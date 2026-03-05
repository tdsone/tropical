from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(help="Tropical — autoregressive mRNA language model")


def _find_latest_checkpoint(checkpoint_dir: str, stage: int) -> str:
    """Find the checkpoint with the highest step count for a given stage."""
    ckpt_dir = Path(checkpoint_dir)
    pattern = f"stage{stage}_step*.pt"
    matches = sorted(ckpt_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No stage {stage} checkpoints found in {ckpt_dir}"
        )
    return str(matches[-1])


@app.command()
def train(
    stage: Annotated[int, typer.Option(help="Training stage (1, 2, or 3)")] = 1,
    data_dir: Annotated[str, typer.Option(help="Path to data directory")] = "./data",
    checkpoint_dir: Annotated[str, typer.Option(help="Path to checkpoint directory")] = "./checkpoints",
    resume_from: Annotated[Optional[str], typer.Option(help="Path to checkpoint to resume from")] = None,
    max_steps: Annotated[int, typer.Option(help="Maximum training steps")] = 100_000,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 32,
    learning_rate: Annotated[float, typer.Option(help="Learning rate")] = 3e-4,
    wandb: Annotated[bool, typer.Option("--wandb/--no-wandb", help="Enable Weights & Biases logging")] = False,
) -> None:
    """Run a single training stage."""
    from tropical.config import TropicalConfig
    from tropical.train import train as run_train

    if stage not in (1, 2, 3):
        raise typer.BadParameter("Stage must be 1, 2, or 3")

    config = TropicalConfig(
        stage=stage,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        wandb_enabled=wandb,
    )
    run_train(config)


@app.command("train-all")
def train_all(
    data_dir: Annotated[str, typer.Option(help="Path to data directory")] = "./data",
    checkpoint_dir: Annotated[str, typer.Option(help="Path to checkpoint directory")] = "./checkpoints",
    max_steps: Annotated[int, typer.Option(help="Maximum training steps per stage")] = 100_000,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 32,
    learning_rate: Annotated[float, typer.Option(help="Learning rate")] = 3e-4,
    wandb: Annotated[bool, typer.Option("--wandb/--no-wandb", help="Enable Weights & Biases logging")] = False,
) -> None:
    """Run all 3 training stages sequentially, auto-chaining checkpoints."""
    from tropical.config import TropicalConfig
    from tropical.train import train as run_train

    for stage in (1, 2, 3):
        resume_from = None
        if stage > 1:
            resume_from = _find_latest_checkpoint(checkpoint_dir, stage - 1)
            print(f"Chaining from checkpoint: {resume_from}")

        config = TropicalConfig(
            stage=stage,
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            resume_from=resume_from,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            wandb_enabled=wandb,
        )
        print(f"\n{'='*60}")
        print(f"  Stage {stage}")
        print(f"{'='*60}\n")
        run_train(config)


@app.command()
def generate(
    checkpoint: Annotated[str, typer.Option(help="Path to model checkpoint")],
    protein: Annotated[Optional[str], typer.Option(help="Protein sequence to condition on")] = None,
    max_length: Annotated[int, typer.Option(help="Maximum generation length")] = 2048,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 1.0,
    top_k: Annotated[Optional[int], typer.Option(help="Top-k sampling")] = None,
) -> None:
    """Generate an mRNA sequence from a checkpoint."""
    import torch

    from tropical.generate import generate as run_generate
    from tropical.tokenizer import AminoAcidTokenizer, NucleotideTokenizer

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    from tropical.model import Tropical

    model = Tropical(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    nt_tok = NucleotideTokenizer()
    aa_tok = AminoAcidTokenizer()

    sequence = run_generate(
        model=model,
        nt_tok=nt_tok,
        aa_tok=aa_tok,
        protein_seq=protein,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
    )
    print(sequence)


if __name__ == "__main__":
    app()
