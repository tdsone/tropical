"""Upload local data/ directory to the Modal volume for remote training.

Usage:
    modal run upload_data_modal.py

Or use the CLI directly:
    modal volume put tropical-data ./data/raw/ /raw/
    modal volume put tropical-data ./data/processed/ /processed/
"""

import os
from pathlib import Path

import modal

app = modal.App("tropical-upload")

data_vol = modal.Volume.from_name("tropical-data", create_if_missing=True)


@app.function(
    image=modal.Image.debian_slim(python_version="3.13"),
    volumes={"/data": data_vol},
    timeout=600,
)
def upload(local_files: dict[str, bytes]):
    """Write files received from local machine into the volume."""
    for rel_path, content in local_files.items():
        dest = Path("/data") / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        print(f"Uploaded: {rel_path} ({len(content):,} bytes)")

    data_vol.commit()
    print("Volume committed.")


@app.local_entrypoint()
def main():
    """Reads all files under local data/ and uploads them to the volume."""
    local_data = Path("data")
    if not local_data.exists():
        print("No data/ directory found locally.")
        return

    files: dict[str, bytes] = {}
    for root, _, filenames in os.walk(local_data):
        for fname in filenames:
            full_path = Path(root) / fname
            rel_path = full_path.relative_to(local_data)
            files[str(rel_path)] = full_path.read_bytes()

    print(f"Uploading {len(files)} files...")
    upload.remote(files)
    print("Done.")
