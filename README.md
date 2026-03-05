# Tropical

Autoregressive mRNA language model conditioned on protein sequence and translation efficiency (TE).

Blog post: https://tsone.notion.site/Autoregressively-generate-cell-type-specific-mRNAs-for-better-vaccines-with-tropical-2e5243755b3f80f1a769e11ef4c3d37b

Given a protein sequence and optional per-cell-type TE targets, Tropical generates an mRNA coding sequence optimized for those conditions. The architecture is a causal transformer decoder with cross-attention to a bidirectional protein encoder and adaptive layer normalization (adaLN) driven by a TE conditioner.

## Architecture

- **Decoder** — 8-layer causal transformer (512-dim, 8 heads) with sinusoidal positional embeddings. Each block has self-attention, cross-attention, and FFN sub-blocks, all gated by adaLN.
- **Protein encoder** — 4-layer bidirectional transformer over amino acid tokens (vocab 25). Output is cross-attended by the decoder.
- **TE conditioner** — Small MLP mapping 78 cell-type TE values (+ presence mask) to per-layer adaLN scale/shift parameters. Zero-initialized so conditioning starts as identity.
- **Tokenizers** — Lookup-table tokenizers for nucleotides (vocab 8: PAD, BOS, EOS, A, C, G, T, U) and amino acids (vocab 25: 20 standard + X/special tokens).

## Training stages

Training uses a 3-stage curriculum with progressive unfreezing:

1. **Stage 1** — mRNA language modeling only. Protein encoder, TE conditioner, and cross-attention are frozen.
2. **Stage 2** — Unfreeze protein encoder and cross-attention. TE conditioner stays frozen.
3. **Stage 3** — All parameters trainable.

## Pretrained weights

Pretrained checkpoints are available on HuggingFace: https://huggingface.co/tdsone/tropical/tree/main

Download a checkpoint and pass it to `--checkpoint` when generating.

## Usage

```bash
# Install
uv sync

# Train a single stage
uv run tropical train --stage 1 --max-steps 50000

# Train all 3 stages (auto-chains checkpoints)
uv run tropical train-all --max-steps 50000

# Generate a sequence conditioned on a protein
uv run tropical generate --checkpoint ./checkpoints/stage3_step50000.pt --protein MVKLT
```

Run `uv run tropical --help` for full CLI options.

### Running the frontend

The frontend is a React app that talks to the Modal backend.

**1. Start the Modal backend:**

```bash
# Requires modal (in dev dependencies: uv sync --extra dev)
# Place a checkpoint in the tropical-checkpoints Modal volume first (see Training on Modal below),
# or upload a local checkpoint:
modal volume put tropical-checkpoints stage3_step50000.pt /stage3_step50000.pt

# Serve the backend (keeps running until interrupted)
uv run modal serve serve_modal.py
```

This prints a set of endpoint URLs. Copy the base URL (e.g. `https://your-app--...modal.run`).

**2. Start the frontend dev server:**

```bash
cd frontend
npm install
VITE_API_URL=https://your-app--...modal.run npm run dev
```

Open http://localhost:5173 in your browser.

### Training on Modal (GPU)

Requires `modal` (in dev dependencies: `uv sync --extra dev`).

```bash
# Upload local data/ to the Modal volume
uv run modal run upload_data_modal.py

# Train all 3 stages on an A100-80GB (auto-chains checkpoints)
uv run modal run train_modal.py --max-steps 50000

# Or train a single stage
uv run modal run train_modal.py --stage 1 --max-steps 50000

# Chain stages manually
uv run modal run train_modal.py --stage 2 --resume-from /checkpoints/stage1_step50000.pt
uv run modal run train_modal.py --stage 3 --resume-from /checkpoints/stage2_step50000.pt
```

Data and checkpoints are persisted on Modal volumes (`tropical-data` and `tropical-checkpoints`), so checkpoint paths use `/data` and `/checkpoints` (the volume mount points).

## Background

- What are known cases where sequence drives cell-type specificity?
    - In the liver, a specific micro-RNA called miRNA-122 is highly abundant. By incorporating the "reverse complement" of its sequence into the 3’ UTR of your payload mRNA, miRNA-122 will bind to the payload and trigger its rapid degradation. Because dendritic cells lack miRNA-122, the mRNA remains stable and produces high protein levels there, while remaining virtually silent in the liver where miRNA-122 is abundant. https://en.wikipedia.org/wiki/MiR-122

- From COVID 19 vaccine report: https://www.ema.europa.eu/en/documents/assessment-report/comirnaty-epar-public-assessment-report_en.pdf
    > Radioactivity was detected in most tissues from the first time point (0.25 h) and results support that injections site and the liver are the major sites of distribution.
    
    > Low levels of radioactivity were detected in most tissues, with the greatest levels in plasma observed 1-4 hours post-dose.

    > Over 48 hours, distribution was mainly observed to liver, adrenal glands, spleen and ovaries, with maximum concentrations observed at 8-48 hours post-dose. Total recovery (% of injected dose) of radiolabeled LNP+modRNA outside the injection site was greatest in the liver (up to 21.5%) and was much less in spleen (≤1.1%), adrenal glands (≤0.1%) and ovaries (≤0.1%).

    - So biodistribution changes over time -> If you don't go for liver delivery, the detargeting the liver is probably no.1 prio; I'm not sure if adrenal glands is just on this list because it was foreign epitopes or if there is a natural preference of LNPs for adrenal glands

- When an LNP enters the bloodstream, it doesn't stay a "naked" nanoparticle. Blood proteins immediately stick to its surface, forming what is called a "protein corona." The most prominent protein that binds to LNPs is Apolipoprotein E (ApoE). With ApoE as a "label" it guides the LNP towards the adrenal glands which require large amounts of cholesterol to make hormones.
- The injection site itself will have a lot of LNP exposure too.
    Muscle Cells (Myocytes): Since the injection is intramuscular, myocytes are the most abundant cells in the immediate vicinity. They take up a massive portion of the LNPs.

    Tissue-Resident Immune Cells: Your muscle tissue is constantly patrolled by local immune cells, primarily macrophages and dendritic cells. These cells are literally designed to sample their environment and swallow foreign particles, so they aggressively gorge on the local LNPs.

    Fibroblasts: These are the structural cells that make up the connective tissue holding your muscle fibers together. They also readily absorb the nearby LNPs.

    Endothelial Cells: These are the cells that line the tiny capillaries and blood vessels running through the muscle.