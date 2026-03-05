"""Upload RiboNN weights from local weights.zip to the Modal volume.

Usage:
    modal run upload_ribonn_weights.py

Or use the CLI directly:
    modal volume put ribonn-weights ./weights/ /
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import modal

app = modal.App("ribonn-upload")

ribonn_vol = modal.Volume.from_name("ribonn-weights", create_if_missing=True)


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={"/weights": ribonn_vol},
    timeout=600,
)
def upload(files: dict[str, bytes]):
    """Write weight files received from local machine into the volume."""
    for rel_path, content in files.items():
        dest = Path("/weights") / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        print(f"  Uploaded: {rel_path} ({len(content):,} bytes)")

    ribonn_vol.commit()
    print("Volume committed.")


@app.local_entrypoint()
def main():
    zip_path = Path("weights.zip")
    if not zip_path.exists():
        print("weights.zip not found in current directory.")
        return

    print(f"Reading {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        # Collect only human/*/state_dict.pth entries
        members = [m for m in zf.infolist() if m.filename.endswith("state_dict.pth") and m.filename.startswith("human/")]

    print(f"Found {len(members)} weight files under human/")

    # Upload in batches to stay within memory limits
    BATCH = 20
    for i in range(0, len(members), BATCH):
        batch_members = members[i : i + BATCH]
        files: dict[str, bytes] = {}
        with zipfile.ZipFile(zip_path) as zf:
            for m in batch_members:
                files[m.filename] = zf.read(m.filename)
        print(f"Uploading batch {i // BATCH + 1}/{(len(members) + BATCH - 1) // BATCH} ({len(files)} files)...")
        upload.remote(files)

    print("Done. All weight files uploaded to ribonn-weights volume.")
