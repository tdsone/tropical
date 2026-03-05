"""Modal FastAPI backend for RiboNN translation efficiency prediction."""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("ribonn-serve")

ribonn_vol = modal.Volume.from_name("ribonn-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "pytorch-lightning==2.1.4",
        "torchmetrics>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "fastapi[standard]>=0.115.0",
        "mlflow>=2.0.0",
    )
    .run_commands("git clone https://github.com/Sanofi-Public/RiboNN /opt/ribonn --depth=1")
    .env({"PYTHONPATH": "/opt/ribonn"})
)

# 78 human cell-type TE targets (order matches the trained model outputs)
HUMAN_TE_COLUMNS = (
    "TE_108T,TE_12T,TE_A2780,TE_A549,TE_BJ,TE_BRx.142,TE_C643,TE_CRL.1634,"
    "TE_Calu.3,TE_Cybrid_Cells,TE_H1.hESC,TE_H1933,TE_H9.hESC,TE_HAP.1,"
    "TE_HCC_tumor,TE_HCC_adjancent_normal,TE_HCT116,TE_HEK293,TE_HEK293T,"
    "TE_HMECs,TE_HSB2,TE_HSPCs,TE_HeLa,TE_HeLa_S3,TE_HepG2,TE_Huh.7.5,"
    "TE_Huh7,TE_K562,TE_Kidney_normal_tissue,TE_LCL,TE_LuCaP.PDX,TE_MCF10A,"
    "TE_MCF10A.ER.Src,TE_MCF7,TE_MD55A3,TE_MDA.MB.231,TE_MM1.S,TE_MOLM.13,"
    "TE_Molt.3,TE_Mutu,TE_OSCC,TE_PANC1,TE_PATU.8902,TE_PC3,TE_PC9,"
    "TE_Primary_CD4._T.cells,TE_Primary_human_bronchial_epithelial_cells,"
    "TE_RD.CCL.136,TE_RPE.1,TE_SH.SY5Y,TE_SUM159PT,TE_SW480TetOnAPC,"
    "TE_T47D,TE_THP.1,TE_U.251,TE_U.343,TE_U2392,TE_U2OS,TE_Vero_6,"
    "TE_WI38,TE_WM902B,TE_WTC.11,TE_ZR75.1,TE_cardiac_fibroblasts,"
    "TE_ccRCC,TE_early_neurons,TE_fibroblast,TE_hESC,TE_human_brain_tumor,"
    "TE_iPSC.differentiated_dopamine_neurons,TE_megakaryocytes,TE_muscle_tissue,"
    "TE_neuronal_precursor_cells,TE_neurons,TE_normal_brain_tissue,"
    "TE_normal_prostate,TE_primary_macrophages,TE_skeletal_muscle"
).split(",")

# Model hyperparameters inferred from the distributed weights
# - num_targets=78: from head.7.weight shape (78, 64)
# - len_after_conv=9: from head.3.weight shape (64, 576) → 576/64=9
# - max_seq_len=13312: unique value that produces len_after_conv=9 with
#     kernel_size=5, conv_padding=0, num_conv_layers=10
RIBONN_CONFIG = {
    "num_targets": 78,
    "len_after_conv": 9,
    "max_seq_len": 13312,
    "pad_5_prime": False,
    "split_utr5_cds_utr3_channels": False,
    "label_codons": True,
    "label_3rd_nt_of_codons": False,
    "label_utr5": False,
    "label_utr3": False,
    "label_splice_sites": False,
    "label_up_probs": False,
    "with_NAs": False,
    "filters": 64,
    "kernel_size": 5,
    "conv_stride": 1,
    "conv_dilation": 1,
    "conv_padding": 0,
    "num_conv_layers": 10,
    "dropout": 0.3,
    "ln_epsilon": 0.007,
    "bn_momentum": 0.9,
    "residual": False,
    "max_shift": 0,
    "symmetric_shift": True,
}


_CODON_TABLE = {
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


def _translate(seq: str) -> tuple[str, int]:
    """Translate seq until stop codon; return (aa_string, stop_codon_end_position).

    stop_codon_end_position is -1 if no stop codon was found.
    """
    aas = []
    for i in range(0, len(seq) - 2, 3):
        aa = _CODON_TABLE.get(seq[i : i + 3], "X")
        if aa == "*":
            return "".join(aas), i + 3
        aas.append(aa)
    return "".join(aas), -1


def _find_cds(sequence: str, protein_seq: str | None) -> tuple[int, int]:
    """Return (utr5_size, cds_size) for the CDS in *sequence*.

    Strategy (in order of preference):
    1. If protein_seq is given, scan every ATG and check whether the translated
       product matches protein_seq (ignoring leading M differences).  The first
       matching ATG wins.
    2. Fallback: first ATG that has a downstream in-frame stop codon.
    3. Last resort: treat the whole sequence as CDS (utr5_size=0).
    """
    seq = sequence.upper().replace("U", "T")

    if protein_seq:
        target = protein_seq.upper().rstrip("*")
        pos = 0
        while True:
            atg_pos = seq.find("ATG", pos)
            if atg_pos == -1:
                break
            translated, stop_end = _translate(seq[atg_pos:])
            if stop_end != -1 and (translated == target or translated == "M" + target.lstrip("M")):
                return atg_pos, stop_end
            pos = atg_pos + 1

    # Fallback: first ATG with in-frame stop
    pos = 0
    while True:
        atg_pos = seq.find("ATG", pos)
        if atg_pos == -1:
            break
        _, stop_end = _translate(seq[atg_pos:])
        if stop_end != -1:
            return atg_pos, stop_end
        pos = atg_pos + 1

    return 0, len(seq)


def _encode_sequence(tx_seq: str, utr5_size: int, cds_size: int, max_seq_len: int):
    """One-hot encode a transcript for RiboNN (pad_5_prime=False, label_codons=True).

    Returns a float32 tensor of shape (5, max_seq_len):
      channels 0-3: one-hot ATCG
      channel   4 : codon-start label (1 at every codon-start position in the CDS)
    """
    import torch

    base_index = {"A": 0, "T": 1, "C": 2, "G": 3, "U": 1}
    x = torch.zeros((5, max_seq_len), dtype=torch.float32)

    for idx, nt in enumerate(tx_seq.upper()):
        if idx >= max_seq_len:
            break
        ch = base_index.get(nt)
        if ch is not None:
            x[ch, idx] = 1.0

    # Mark every codon start (positions utr5_size, utr5_size+3, …, utr5_size+cds_size-3)
    start = utr5_size
    stop = start + cds_size - 3
    for idx in range(start, min(stop + 1, max_seq_len), 3):
        x[4, idx] = 1.0

    return x


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/weights": ribonn_vol},
    scaledown_window=15 * 60,
)
@modal.concurrent(max_inputs=10)
class RiboNNInference:
    @modal.enter()
    def load_models(self):
        import sys
        import torch

        sys.path.insert(0, "/opt/ribonn")
        from src.model import RiboNN  # noqa: PLC0415

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weight_dir = Path("/weights/human")
        run_ids = sorted(d.name for d in weight_dir.iterdir() if d.is_dir())

        self.models = []
        for run_id in run_ids[:5]:
            ckpt_path = weight_dir / run_id / "state_dict.pth"
            model = RiboNN(**RIBONN_CONFIG)
            state = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            self.models.append(model)

        print(f"Loaded {len(self.models)} RiboNN models on {self.device}")

    @modal.asgi_app()
    def web(self):
        import torch
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        api = FastAPI()
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @api.get("/health")
        def health():
            return {"status": "ok", "n_models": len(self.models), "device": str(self.device)}

        @api.get("/columns")
        def columns():
            return {"columns": HUMAN_TE_COLUMNS}

        @api.post("/predict")
        def predict(request: dict):
            sequence: str = request.get("sequence", "")
            protein_seq: str | None = request.get("protein_seq")

            sequence = sequence.upper().replace("U", "T")
            max_seq_len = RIBONN_CONFIG["max_seq_len"]

            if len(sequence) > max_seq_len:
                sequence = sequence[:max_seq_len]

            utr5_size, cds_size = _find_cds(sequence, protein_seq)

            x = _encode_sequence(
                sequence,
                utr5_size=utr5_size,
                cds_size=cds_size,
                max_seq_len=max_seq_len,
            )
            x = x.unsqueeze(0).to(self.device)  # (1, 5, max_seq_len)

            all_preds = []
            with torch.no_grad():
                for model in self.models:
                    pred = model(x)  # (1, 78)
                    all_preds.append(pred)

            mean_pred = torch.stack(all_preds, dim=0).mean(dim=0).squeeze(0)  # (78,)
            values = mean_pred.cpu().tolist()

            return {
                "predictions": {col: val for col, val in zip(HUMAN_TE_COLUMNS, values)},
                "columns": HUMAN_TE_COLUMNS,
                "values": values,
                "utr5_size": utr5_size,
                "cds_size": cds_size,
            }

        return api
