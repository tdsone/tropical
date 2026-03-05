from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tropical.config import TropicalConfig


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with fused QKV projection."""

    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.qkv_proj = nn.Linear(emb_dim, 3 * emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.unflatten(-1, (self.n_heads, self.head_dim)).transpose(1, 2)
        k = k.unflatten(-1, (self.n_heads, self.head_dim)).transpose(1, 2)
        v = v.unflatten(-1, (self.n_heads, self.head_dim)).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = attn.transpose(1, 2).flatten(-2)  # (B, T, C)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Multi-head cross-attention: Q from decoder, K/V from encoder context."""

    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.kv_proj = nn.Linear(emb_dim, 2 * emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) decoder hidden states
            context: (B, S, C) encoder output
            context_pad_mask: (B, S) True where context is padding
        """
        B, T, C = x.shape
        S = context.shape[1]

        q = self.q_proj(x).unflatten(-1, (self.n_heads, self.head_dim)).transpose(1, 2)
        kv = self.kv_proj(context)
        k, v = kv.chunk(2, dim=-1)
        k = k.unflatten(-1, (self.n_heads, self.head_dim)).transpose(1, 2)
        v = v.unflatten(-1, (self.n_heads, self.head_dim)).transpose(1, 2)

        # Build attention mask: (B, 1, 1, S) — True positions are masked out
        attn_mask = None
        if context_pad_mask is not None:
            # SDPA expects: True = attend, so invert padding mask
            attn_mask = ~context_pad_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            attn_mask = attn_mask.expand(B, self.n_heads, T, S)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        # When all context positions are masked, softmax produces NaN.
        # Replace with zeros — no protein signal is the correct behavior.
        attn = torch.nan_to_num(attn, nan=0.0)
        out = attn.transpose(1, 2).flatten(-2)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Feed-forward network
# ---------------------------------------------------------------------------


class FFN(nn.Module):
    """Position-wise feed-forward with GELU and 4x expansion."""

    def __init__(self, emb_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, 4 * emb_dim)
        self.fc2 = nn.Linear(4 * emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


# ---------------------------------------------------------------------------
# Adaptive Layer Norm + TE Conditioner
# ---------------------------------------------------------------------------


class AdaLN(nn.Module):
    """Adaptive Layer Normalization: LN(x) * gamma + beta."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(emb_dim, elementwise_affine=False)

    def forward(
        self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
            gamma: (B, 1, C) — scale
            beta: (B, 1, C) — shift
        """
        return gamma * self.ln(x) + beta


class TEConditioner(nn.Module):
    """Maps TE values + mask to per-layer adaLN parameters.

    Input: (B, 78) TE values + (B, 78) binary mask → concatenated (B, 156).
    Output: per-layer (gamma, beta) for self-attn, cross-attn, and FFN adaLN.
    That's n_layers * 3 sub-blocks * 2 (gamma, beta) * emb_dim parameters.
    """

    def __init__(self, n_te_conditions: int, emb_dim: int, n_layers: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        # 3 sub-blocks per layer (self-attn, cross-attn, FFN), each needs gamma + beta
        self.n_params = n_layers * 3 * 2 * emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(n_te_conditions * 2, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, self.n_params),
        )

        # Zero-init final layer so adaLN is identity at init
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, te_values: torch.Tensor, te_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            te_values: (B, 78) — translation efficiency values
            te_mask: (B, 78) — 1.0 where present, 0.0 where missing
        Returns:
            (B, n_layers, 3, 2, emb_dim) — per-layer adaLN params
        """
        x = torch.cat([te_values * te_mask, te_mask], dim=-1)  # (B, 156)
        params = self.mlp(x)  # (B, n_params)
        params = params.view(-1, self.n_layers, 3, 2, self.emb_dim)
        # Add identity: gamma = 1 + delta_gamma, beta = 0 + delta_beta
        # Since zero-init, delta starts at 0. We add 1 to gamma channel.
        params[:, :, :, 0, :] = params[:, :, :, 0, :] + 1.0  # gamma
        return params


# ---------------------------------------------------------------------------
# Protein Encoder
# ---------------------------------------------------------------------------


class ProteinEncoderBlock(nn.Module):
    """Bidirectional transformer block for protein encoder."""

    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(
            emb_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffn = FFN(emb_dim, dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x


class ProteinEncoder(nn.Module):
    """Bidirectional transformer encoding amino acid sequences."""

    def __init__(
        self,
        n_amino_acids: int,
        emb_dim: int,
        protein_block_size: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(n_amino_acids, emb_dim)
        self.pos_emb = nn.Embedding(protein_block_size, emb_dim)
        self.blocks = nn.ModuleList(
            [ProteinEncoderBlock(emb_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(emb_dim)

    def forward(
        self, protein_ids: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            protein_ids: (B, S) amino acid token IDs
            pad_mask: (B, S) True where padding
        Returns:
            (B, S, emb_dim)
        """
        B, S = protein_ids.shape
        positions = torch.arange(S, device=protein_ids.device).unsqueeze(0)
        x = self.token_emb(protein_ids) + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)

        return self.ln_final(x)


# ---------------------------------------------------------------------------
# Decoder Transformer Block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Decoder block: AdaLN → self-attn, AdaLN → cross-attn, AdaLN → FFN."""

    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Self-attention
        self.adaln_sa = AdaLN(emb_dim)
        self.self_attn = CausalSelfAttention(emb_dim, n_heads, dropout)

        # Cross-attention
        self.adaln_ca = AdaLN(emb_dim)
        self.cross_attn = CrossAttention(emb_dim, n_heads, dropout)

        # Feed-forward
        self.adaln_ff = AdaLN(emb_dim)
        self.ffn = FFN(emb_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        adaln_params: torch.Tensor | None = None,
        protein_ctx: torch.Tensor | None = None,
        protein_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
            adaln_params: (B, 3, 2, C) — params for this layer's 3 sub-blocks
            protein_ctx: (B, S, C) from protein encoder, or None
            protein_pad_mask: (B, S) True where padding
        """
        if adaln_params is not None:
            sa_gamma, sa_beta = adaln_params[:, 0, 0:1, :], adaln_params[:, 0, 1:2, :]
            ca_gamma, ca_beta = adaln_params[:, 1, 0:1, :], adaln_params[:, 1, 1:2, :]
            ff_gamma, ff_beta = adaln_params[:, 2, 0:1, :], adaln_params[:, 2, 1:2, :]
        else:
            # Default identity: gamma=1, beta=0
            ones = torch.ones(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
            zeros = torch.zeros_like(ones)
            sa_gamma, sa_beta = ones, zeros
            ca_gamma, ca_beta = ones, zeros
            ff_gamma, ff_beta = ones, zeros

        # 1. Self-attention
        x = x + self.self_attn(self.adaln_sa(x, sa_gamma, sa_beta))

        # 2. Cross-attention (skip when no protein context)
        if protein_ctx is not None:
            x = x + self.cross_attn(
                self.adaln_ca(x, ca_gamma, ca_beta),
                protein_ctx,
                context_pad_mask=protein_pad_mask,
            )

        # 3. Feed-forward
        x = x + self.ffn(self.adaln_ff(x, ff_gamma, ff_beta))

        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class Tropical(nn.Module):
    """Autoregressive mRNA language model conditioned on protein (cross-attention)
    and translation efficiency (adaLN)."""

    def __init__(self, config: TropicalConfig) -> None:
        super().__init__()
        self.config = config
        emb_dim = config.emb_dim

        # Token + positional embeddings (sinusoidal)
        self.token_emb = nn.Embedding(config.vocab_size, emb_dim)

        position = torch.arange(config.block_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim)
        )
        pos_emb = torch.zeros(config.block_size, emb_dim)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pos_emb)

        # Protein encoder
        self.protein_encoder = ProteinEncoder(
            n_amino_acids=config.n_amino_acids,
            emb_dim=emb_dim,
            protein_block_size=config.protein_block_size,
            n_layers=config.protein_encoder_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

        # TE conditioner
        self.te_conditioner = TEConditioner(
            n_te_conditions=config.n_te_conditions,
            emb_dim=emb_dim,
            n_layers=config.n_layers,
        )

        # Decoder blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(emb_dim, config.n_heads, config.dropout)
                for _ in range(config.n_layers)
            ]
        )

        self.ln_final = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "te_conditioner" in name:
                continue  # TEConditioner has its own init
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        idx: torch.Tensor,
        protein_ids: torch.Tensor | None = None,
        protein_pad_mask: torch.Tensor | None = None,
        te_values: torch.Tensor | None = None,
        te_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            idx: (B, T) nucleotide token IDs
            protein_ids: (B, S) amino acid token IDs, or None
            protein_pad_mask: (B, S) True where protein is padding
            te_values: (B, 78) translation efficiency values
            te_mask: (B, 78) 1.0 where TE is present, 0.0 where missing
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_emb(idx)
        x = tok_emb + self.pos_emb[:T].unsqueeze(0)

        # Encode protein context (skip when entire batch is padding)
        protein_ctx = None
        has_protein = protein_ids is not None and (
            protein_pad_mask is None or not protein_pad_mask.all()
        )
        if has_protein:
            protein_ctx = self.protein_encoder(protein_ids, pad_mask=protein_pad_mask)

        # Compute adaLN parameters from TE
        adaln_all = None
        if te_values is not None and te_mask is not None:
            adaln_all = self.te_conditioner(te_values, te_mask)
            # adaln_all: (B, n_layers, 3, 2, emb_dim)

        for i, block in enumerate(self.blocks):
            adaln_params = adaln_all[:, i] if adaln_all is not None else None
            x = block(
                x,
                adaln_params=adaln_params,
                protein_ctx=protein_ctx,
                protein_pad_mask=protein_pad_mask,
            )

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
