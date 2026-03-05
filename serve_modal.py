"""Modal FastAPI backend for Tropical mRNA generation."""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("tropical-serve")

ckpt_vol = modal.Volume.from_name("tropical-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.10.0",
        "fastapi[standard]>=0.115.0",
    )
    .add_local_python_source("tropical")
)


@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/checkpoints": ckpt_vol},
    scaledown_window=15 * 60,
)
@modal.concurrent(max_inputs=10)
class Inference:
    @modal.enter()
    def load_model(self):
        import torch

        from tropical.model import Tropical
        from tropical.tokenizer import AminoAcidTokenizer, NucleotideTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Find latest stage-3 checkpoint
        matches = sorted(Path("/checkpoints").glob("stage3_step*.pt"))
        if not matches:
            # Fall back to any available checkpoint
            matches = sorted(Path("/checkpoints").glob("stage*_step*.pt"))
        if not matches:
            raise FileNotFoundError("No checkpoints found in /checkpoints")

        ckpt_path = matches[-1]
        print(f"Loading checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = ckpt["config"]

        self.model = Tropical(config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.nt_tok = NucleotideTokenizer()
        self.aa_tok = AminoAcidTokenizer()
        self.config = config

        print(f"Model loaded on {self.device} from {ckpt_path}")

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {"status": "ok", "device": str(self.device)}

    @modal.fastapi_endpoint(method="GET")
    def te_columns(self):
        from tropical.config import TE_COLUMNS

        return {"columns": TE_COLUMNS}

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: dict):
        import torch

        from tropical.generate import generate as run_generate

        protein_seq = request.get("protein_seq")
        te_values_list = request.get("te_values")
        te_mask_list = request.get("te_mask")
        max_length = request.get("max_length", 2048)
        temperature = request.get("temperature", 1.0)
        top_k = request.get("top_k")

        te_values = None
        te_mask = None
        if te_values_list is not None and te_mask_list is not None:
            te_values = torch.tensor(te_values_list, dtype=torch.float32)
            te_mask = torch.tensor(te_mask_list, dtype=torch.float32)

        sequence = run_generate(
            model=self.model,
            nt_tok=self.nt_tok,
            aa_tok=self.aa_tok,
            protein_seq=protein_seq,
            te_values=te_values,
            te_mask=te_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
        )

        return {"sequence": sequence, "length": len(sequence)}
