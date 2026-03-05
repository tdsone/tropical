import modal

app = modal.App("tropical")

data_vol = modal.Volume.from_name("tropical-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("tropical-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.10.0",
        "pandas>=3.0.1",
        "pyarrow>=19.0.0",
        "openpyxl>=3.1.5",
        "wandb>=0.25.0",
    )
    .add_local_python_source("tropical")
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=86400,
    volumes={"/data": data_vol, "/checkpoints": ckpt_vol},
    secrets=[modal.Secret.from_name("custom-secret")],
)
def train_remote(
    stage: int = 1,
    resume_from: str | None = None,
    max_steps: int = 100_000,
):
    from pathlib import Path

    from tropical.config import TropicalConfig
    from tropical.train import train

    # Auto-find latest checkpoint from the current or previous stage
    if resume_from is None and stage > 1:
        # First try resuming within the same stage (interrupted run)
        for s in (stage, stage - 1):
            pattern = f"stage{s}_step*.pt"
            matches = sorted(Path("/checkpoints").glob(pattern))
            if matches:
                resume_from = str(matches[-1])
                print(f"Auto-resuming from checkpoint: {resume_from}")
                break
        if resume_from is None:
            raise FileNotFoundError(
                f"No stage {stage} or stage {stage - 1} checkpoints found in /checkpoints"
            )

    config = TropicalConfig(
        stage=stage,
        data_dir="/data",
        checkpoint_dir="/checkpoints",
        resume_from=resume_from,
        max_steps=max_steps,
        wandb_enabled=True,
    )
    train(config)

    # Commit volume changes so checkpoints persist
    ckpt_vol.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=86400,
    volumes={"/data": data_vol, "/checkpoints": ckpt_vol},
    secrets=[modal.Secret.from_name("custom-secret")],
)
def find_latest_checkpoint(stage: int) -> str | None:
    """Find the latest checkpoint for a given stage on the volume."""
    from pathlib import Path

    pattern = f"stage{stage}_step*.pt"
    matches = sorted(Path("/checkpoints").glob(pattern))
    return str(matches[-1]) if matches else None


@app.local_entrypoint()
def main(
    stage: int = 0,
    resume_from: str | None = None,
    max_steps: int = 100_000,
):
    if stage == 0:
        # Train all 3 stages, each as a separate remote call so
        # each stage gets its own full 24h timeout.
        for s in (1, 2, 3):
            print(f"\n{'=' * 60}")
            print(f"  Stage {s}")
            print(f"{'=' * 60}\n")
            train_remote.remote(stage=s, max_steps=max_steps)
    else:
        train_remote.remote(stage=stage, resume_from=resume_from, max_steps=max_steps)
