from __future__ import annotations

import torch
import torch.nn.functional as F

from tropical.model import Tropical
from tropical.tokenizer import AminoAcidTokenizer, NucleotideTokenizer


@torch.no_grad()
def generate(
    model: Tropical,
    nt_tok: NucleotideTokenizer,
    aa_tok: AminoAcidTokenizer,
    protein_seq: str | None = None,
    te_values: torch.Tensor | None = None,
    te_mask: torch.Tensor | None = None,
    max_length: int = 2048,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    """Autoregressive mRNA sequence generation.

    Args:
        model: Trained Tropical model (in eval mode).
        nt_tok: Nucleotide tokenizer.
        aa_tok: Amino acid tokenizer.
        protein_seq: Optional protein sequence to condition on.
        te_values: Optional (78,) TE values tensor.
        te_mask: Optional (78,) TE mask tensor (1.0 where specified).
        max_length: Maximum generation length.
        temperature: Sampling temperature (1.0 = normal, <1.0 = sharper).
        top_k: If set, sample from top-k tokens only.

    Returns:
        Generated nucleotide sequence string.
    """
    model.eval()
    device = next(model.parameters()).device
    config = model.config

    # Prepare protein context
    protein_ids = None
    protein_pad_mask = None
    if protein_seq is not None:
        prot_ids = aa_tok.encode(protein_seq)
        if len(prot_ids) > config.protein_block_size:
            prot_ids = prot_ids[: config.protein_block_size]
        else:
            prot_ids = prot_ids + [aa_tok.pad_id] * (config.protein_block_size - len(prot_ids))
        protein_ids = torch.tensor([prot_ids], dtype=torch.long, device=device)
        protein_pad_mask = protein_ids == aa_tok.pad_id

    # Prepare TE conditioning
    te_vals_batch = None
    te_mask_batch = None
    if te_values is not None and te_mask is not None:
        te_vals_batch = te_values.unsqueeze(0).to(device)
        te_mask_batch = te_mask.unsqueeze(0).to(device)

    # Start with BOS token
    idx = torch.tensor([[nt_tok.bos_id]], dtype=torch.long, device=device)

    for _ in range(max_length - 1):
        # Truncate to block_size if needed
        idx_cond = idx if idx.shape[1] <= config.block_size else idx[:, -config.block_size :]

        logits = model(
            idx=idx_cond,
            protein_ids=protein_ids,
            protein_pad_mask=protein_pad_mask,
            te_values=te_vals_batch,
            te_mask=te_mask_batch,
        )

        # Get logits for last position
        logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

        # Stop at EOS
        if next_token.item() == nt_tok.eos_id:
            break

    return nt_tok.decode(idx[0].tolist())
