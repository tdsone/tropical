from __future__ import annotations


class NucleotideTokenizer:
    """Lookup-table tokenizer for nucleotide sequences.

    Vocabulary: [PAD]=0, [BOS]=1, [EOS]=2, A=3, C=4, G=5, T=6, U=7
    U is normalized to T during encoding.
    """

    VOCAB = ["[PAD]", "[BOS]", "[EOS]", "A", "C", "G", "T", "U"]

    def __init__(self) -> None:
        self.stoi = {tok: i for i, tok in enumerate(self.VOCAB)}
        self.itos = {i: tok for i, tok in enumerate(self.VOCAB)}
        self.pad_id = self.stoi["[PAD]"]
        self.bos_id = self.stoi["[BOS]"]
        self.eos_id = self.stoi["[EOS]"]

    @property
    def vocab_size(self) -> int:
        return len(self.VOCAB)

    def encode(self, seq: str) -> list[int]:
        """Encode nucleotide sequence with BOS/EOS. Normalizes U -> T."""
        ids = [self.bos_id]
        for ch in seq.upper():
            if ch == "U":
                ch = "T"
            if ch in self.stoi:
                ids.append(self.stoi[ch])
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to nucleotide string (strips special tokens)."""
        chars = []
        for i in ids:
            tok = self.itos.get(i, "")
            if tok in ("[PAD]", "[BOS]", "[EOS]"):
                continue
            chars.append(tok)
        return "".join(chars)


class AminoAcidTokenizer:
    """Lookup-table tokenizer for amino acid sequences.

    Vocabulary: [PAD]=0, [BOS]=1, [EOS]=2, [MASK]=3, A=4, C=5, ..., Y=23, X=24
    X represents unknown amino acid. * (stop codon) is stripped.
    """

    _AA = "ACDEFGHIKLMNPQRSTVWY"
    VOCAB = ["[PAD]", "[BOS]", "[EOS]", "[MASK]"] + list(_AA) + ["X"]

    def __init__(self) -> None:
        self.stoi = {tok: i for i, tok in enumerate(self.VOCAB)}
        self.itos = {i: tok for i, tok in enumerate(self.VOCAB)}
        self.pad_id = self.stoi["[PAD]"]
        self.bos_id = self.stoi["[BOS]"]
        self.eos_id = self.stoi["[EOS]"]
        self.mask_id = self.stoi["[MASK]"]
        self.x_id = self.stoi["X"]

    @property
    def vocab_size(self) -> int:
        return len(self.VOCAB)

    def encode(self, seq: str) -> list[int]:
        """Encode amino acid sequence with BOS/EOS. Unknown AAs map to X."""
        ids = [self.bos_id]
        for ch in seq.upper():
            if ch == "*":
                continue
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            else:
                ids.append(self.x_id)
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to amino acid string (strips special tokens)."""
        chars = []
        for i in ids:
            tok = self.itos.get(i, "")
            if tok in ("[PAD]", "[BOS]", "[EOS]", "[MASK]"):
                continue
            chars.append(tok)
        return "".join(chars)
