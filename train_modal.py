import modal

app = modal.App("tropical")

data_vol = modal.Volume.from_name("tropical-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("tropical-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch>=2.10.0", "pandas>=3.0.1", "pyarrow>=19.0.0", "openpyxl>=3.1.5")
    .add_local_python_source("tropical")
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=86400,
    volumes={"/data": data_vol, "/checkpoints": ckpt_vol},
)
def train_remote(
    stage: int = 1,
    resume_from: str | None = None,
    max_steps: int = 100_000,
):
    from tropical.config import TropicalConfig
    from tropical.train import train

    config = TropicalConfig(
        stage=stage,
        data_dir="/data",
        checkpoint_dir="/checkpoints",
        resume_from=resume_from,
        max_steps=max_steps,
    )
    train(config)

    # Commit volume changes so checkpoints persist
    ckpt_vol.commit()


@app.local_entrypoint()
def main(
    stage: int = 1,
    resume_from: str | None = None,
    max_steps: int = 100_000,
):
    train_remote.remote(stage=stage, resume_from=resume_from, max_steps=max_steps)
