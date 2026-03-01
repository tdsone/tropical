from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tropical.config import TropicalConfig, TE_COLUMNS
from tropical.tokenizer import AminoAcidTokenizer, NucleotideTokenizer

# Standard genetic code for translating CDS → protein
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _translate_cds(cds: str) -> str:
    """Translate a CDS nucleotide sequence to amino acid sequence."""
    cds = cds.upper().replace("U", "T")
    aas = []
    for i in range(0, len(cds) - 2, 3):
        codon = cds[i : i + 3]
        aa = CODON_TABLE.get(codon, "X")
        if aa == "*":
            break
        aas.append(aa)
    return "".join(aas)


def _pad_or_truncate(ids: list[int], length: int, pad_id: int) -> list[int]:
    if len(ids) >= length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


class TranscriptDataset(Dataset):
    """Dataset for all training stages.

    Stage 1: transcript sequences only (synthetic + Ensembl)
    Stage 2: + protein sequences
    Stage 3: + real TE values from Excel
    """

    def __init__(
        self,
        config: TropicalConfig,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        self.config = config
        self.nt_tok = NucleotideTokenizer()
        self.aa_tok = AminoAcidTokenizer()

        records = self._load_records(config)

        # Deterministic train/val split (90/10)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(records))
        split_idx = int(0.9 * len(records))
        if split == "train":
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]

        self.records = [records[i] for i in indices]

    def _load_records(self, config: TropicalConfig) -> list[dict]:
        """Load and unify data from all sources for the given stage."""
        data_dir = Path(config.data_dir)
        records: list[dict] = []

        # --- Ensembl parquet (stages 1, 2, 3) ---
        ensembl_path = data_dir / "raw" / "ensembl_transcripts_homo_sapiens.parquet"
        if ensembl_path.exists():
            df = pd.read_parquet(ensembl_path)
            for _, row in df.iterrows():
                rec = {
                    "transcript": str(row["transcript_sequence"]),
                    "protein": None,
                    "te_values": None,
                }
                if config.stage >= 2:
                    prot = row.get("protein_sequence")
                    if pd.notna(prot) and str(prot).strip():
                        rec["protein"] = str(prot).rstrip("*")

                records.append(rec)

        # --- Real TE Excel (stage 3 only) ---
        if config.stage >= 3:
            te_path = data_dir / "raw" / "41587_2025_2712_MOESM3_ESM.xlsx"
            if te_path.exists():
                df = pd.read_excel(te_path)
                for _, row in df.iterrows():
                    tx_seq = str(row["tx_sequence"]) if pd.notna(row.get("tx_sequence")) else None
                    if not tx_seq:
                        continue

                    # Extract protein from CDS
                    protein = None
                    utr5_size = row.get("utr5_size")
                    cds_size = row.get("cds_size")
                    if pd.notna(utr5_size) and pd.notna(cds_size):
                        utr5 = int(utr5_size)
                        cds_len = int(cds_size)
                        cds = tx_seq[utr5 : utr5 + cds_len]
                        if len(cds) >= 3:
                            protein = _translate_cds(cds)
                            if not protein:
                                protein = None

                    # TE values
                    te_vals = np.zeros(config.n_te_conditions, dtype=np.float32)
                    te_mask = np.zeros(config.n_te_conditions, dtype=np.float32)
                    for i, col in enumerate(TE_COLUMNS):
                        val = row.get(col)
                        if pd.notna(val):
                            te_vals[i] = float(val)
                            te_mask[i] = 1.0

                    records.append({
                        "transcript": tx_seq,
                        "protein": protein,
                        "te_values": te_vals,
                        "te_mask": te_mask,
                    })

        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        config = self.config

        # --- Nucleotide input/labels ---
        nt_ids = self.nt_tok.encode(rec["transcript"])
        # Truncate to block_size + 1 (input + next-token label)
        nt_ids = _pad_or_truncate(nt_ids, config.block_size + 1, self.nt_tok.pad_id)
        input_ids = torch.tensor(nt_ids[:-1], dtype=torch.long)
        labels = torch.tensor(nt_ids[1:], dtype=torch.long)
        # Mask padding positions in labels
        labels[input_ids == self.nt_tok.pad_id] = -100
        # Also mask the position after the last real token
        pad_positions = (labels == self.nt_tok.pad_id)
        labels[pad_positions] = -100

        # --- Protein ---
        if rec["protein"] is not None:
            prot_ids = self.aa_tok.encode(rec["protein"])
            prot_ids = _pad_or_truncate(prot_ids, config.protein_block_size, self.aa_tok.pad_id)
            protein_ids = torch.tensor(prot_ids, dtype=torch.long)
            protein_pad_mask = protein_ids == self.aa_tok.pad_id
        else:
            # No protein: all padding (cross-attention will be masked out)
            protein_ids = torch.zeros(config.protein_block_size, dtype=torch.long)
            protein_pad_mask = torch.ones(config.protein_block_size, dtype=torch.bool)

        # --- TE values ---
        if rec.get("te_values") is not None:
            te_values = torch.tensor(rec["te_values"], dtype=torch.float32)
            te_mask = torch.tensor(rec["te_mask"], dtype=torch.float32)
        else:
            te_values = torch.zeros(config.n_te_conditions, dtype=torch.float32)
            te_mask = torch.zeros(config.n_te_conditions, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "protein_ids": protein_ids,
            "protein_pad_mask": protein_pad_mask,
            "te_values": te_values,
            "te_mask": te_mask,
        }
