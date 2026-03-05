"""Build transcript + protein dataset from Ensembl BioMart.

Fetches cDNA, CDS, and peptide sequences per chromosome via pybiomart,
merges them on transcript ID, and writes one parquet file per species.

Falls back to the Ensembl REST API when pybiomart queries fail.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Species configuration
# ---------------------------------------------------------------------------

SPECIES_CONFIG: dict[str, dict] = {
    "homo_sapiens": {
        "dataset": "hsapiens_gene_ensembl",
        "chromosomes": [str(i) for i in range(1, 23)] + ["X", "Y", "MT"],
    },
    "mus_musculus": {
        "dataset": "mmusculus_gene_ensembl",
        "chromosomes": [str(i) for i in range(1, 20)] + ["X", "Y", "MT"],
    },
}

# ---------------------------------------------------------------------------
# pybiomart helpers
# ---------------------------------------------------------------------------

_SEQUENCE_ATTRS: dict[str, list[str]] = {
    "cdna": ["ensembl_transcript_id", "cdna"],
    "coding": ["ensembl_transcript_id", "coding"],
    "peptide": ["ensembl_transcript_id", "peptide"],
}

# pybiomart returns display names — map them back to our attribute names
_DISPLAY_TO_ATTR: dict[str, str] = {
    "Transcript stable ID": "ensembl_transcript_id",
    "cDNA sequences": "cdna",
    "Coding sequence": "coding",
    "Peptide": "peptide",
}


def query_biomart_sequences(
    dataset_name: str,
    seq_type: str,
    chromosome: str,
    retries: int = 3,
) -> pd.DataFrame:
    """Query BioMart for one sequence type on one chromosome.

    Returns a DataFrame with columns ['ensembl_transcript_id', <seq_type>].
    """
    from pybiomart import Dataset

    attrs = _SEQUENCE_ATTRS[seq_type]
    filters = {
        "transcript_biotype": "protein_coding",
        "chromosome_name": chromosome,
    }

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            ds = Dataset(name=dataset_name, host="http://www.ensembl.org")
            df = ds.query(attributes=attrs, filters=filters)
            if df is None or df.empty:
                return pd.DataFrame(columns=attrs)
            # Normalize column names (pybiomart returns display names)
            df = df.rename(columns=_DISPLAY_TO_ATTR)
            # Drop rows with missing sequences
            df = df.dropna(subset=[seq_type])
            df[seq_type] = df[seq_type].astype(str)
            return df
        except Exception as exc:
            last_exc = exc
            wait = 2**attempt
            log.warning(
                "BioMart query %s chr%s attempt %d/%d failed: %s. Retrying in %ds...",
                seq_type,
                chromosome,
                attempt,
                retries,
                exc,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"BioMart query {seq_type} chr{chromosome} failed after {retries} attempts"
    ) from last_exc


def fetch_chromosome(
    dataset_name: str,
    chromosome: str,
) -> pd.DataFrame:
    """Fetch and merge cDNA, CDS, and peptide for one chromosome via pybiomart."""
    log.info("Querying BioMart for chr%s ...", chromosome)

    cdna_df = query_biomart_sequences(dataset_name, "cdna", chromosome)
    coding_df = query_biomart_sequences(dataset_name, "coding", chromosome)
    peptide_df = query_biomart_sequences(dataset_name, "peptide", chromosome)

    merged = (
        cdna_df.merge(coding_df, on="ensembl_transcript_id", how="inner")
        .merge(peptide_df, on="ensembl_transcript_id", how="inner")
    )

    log.info(
        "chr%s: %d cdna, %d coding, %d peptide -> %d merged",
        chromosome,
        len(cdna_df),
        len(coding_df),
        len(peptide_df),
        len(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# CDS start computation
# ---------------------------------------------------------------------------


def compute_cds_start(cdna: str, coding: str) -> int | None:
    """Return 1-based CDS start position within the cDNA.

    Tries exact substring match first.  If that fails, strips a trailing
    stop codon from the coding sequence (BioMart sometimes includes it in
    cdna but not coding, or vice-versa) and retries.
    """
    idx = cdna.find(coding)
    if idx >= 0:
        return idx + 1

    # Strip trailing stop codon from coding and retry
    stop_codons = ("TAA", "TAG", "TGA", "taa", "tag", "tga")
    if len(coding) >= 3 and coding[-3:] in stop_codons:
        coding_stripped = coding[:-3]
        idx = cdna.find(coding_stripped)
        if idx >= 0:
            return idx + 1

    # Try adding stop codons to coding
    for stop in ("TAA", "TAG", "TGA"):
        idx = cdna.find(coding + stop)
        if idx >= 0:
            return idx + 1

    return None


# ---------------------------------------------------------------------------
# Ensembl REST API fallback
# ---------------------------------------------------------------------------

REST_BASE = "https://rest.ensembl.org"
REST_BATCH_SIZE = 50
REST_MIN_INTERVAL = 1.0 / 14  # ~14 req/s


def _rest_post_sequences(
    transcript_ids: list[str],
    seq_type: str,
) -> dict[str, str]:
    """POST to /sequence/id for a batch of transcript IDs.

    seq_type is one of: 'cdna', 'cds', 'protein'.
    Returns {transcript_id: sequence}.
    """
    url = f"{REST_BASE}/sequence/id"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {"ids": transcript_ids, "type": seq_type}

    for attempt in range(5):
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 2))
            log.warning("REST 429 — retrying in %.1fs", retry_after)
            time.sleep(retry_after)
            continue
        resp.raise_for_status()
        break
    else:
        resp.raise_for_status()

    results: dict[str, str] = {}
    for entry in resp.json():
        tid = entry.get("id", "")
        seq = entry.get("seq", "")
        if tid and seq:
            results[tid] = seq
    return results


def _get_transcript_ids_for_chromosome(
    dataset_name: str,
    chromosome: str,
) -> list[str]:
    """Lightweight pybiomart query: just transcript IDs (no sequences)."""
    from pybiomart import Dataset

    ds = Dataset(name=dataset_name, host="http://www.ensembl.org")
    df = ds.query(
        attributes=["ensembl_transcript_id"],
        filters={
            "transcript_biotype": "protein_coding",
            "chromosome_name": chromosome,
        },
    )
    if df is None or df.empty:
        return []
    return df.iloc[:, 0].dropna().unique().tolist()


def fetch_sequences_rest_api(
    dataset_name: str,
    chromosome: str,
) -> pd.DataFrame:
    """Fetch cDNA, CDS, and peptide via the Ensembl REST API."""
    log.info("REST fallback for chr%s — fetching transcript IDs...", chromosome)
    transcript_ids = _get_transcript_ids_for_chromosome(dataset_name, chromosome)
    log.info("chr%s: %d protein-coding transcript IDs", chromosome, len(transcript_ids))

    if not transcript_ids:
        return pd.DataFrame(columns=["ensembl_transcript_id", "cdna", "coding", "peptide"])

    # Map REST seq types to our column names
    type_map = {"cdna": "cdna", "cds": "coding", "protein": "peptide"}
    all_seqs: dict[str, dict[str, str]] = {col: {} for col in type_map.values()}

    for rest_type, col_name in type_map.items():
        for i in range(0, len(transcript_ids), REST_BATCH_SIZE):
            batch = transcript_ids[i : i + REST_BATCH_SIZE]
            t0 = time.monotonic()
            try:
                results = _rest_post_sequences(batch, rest_type)
                all_seqs[col_name].update(results)
            except Exception:
                log.warning(
                    "REST batch %s chr%s offset %d failed, skipping",
                    rest_type,
                    chromosome,
                    i,
                    exc_info=True,
                )
            elapsed = time.monotonic() - t0
            if elapsed < REST_MIN_INTERVAL:
                time.sleep(REST_MIN_INTERVAL - elapsed)

        log.info("chr%s %s: %d sequences fetched via REST", chromosome, rest_type, len(all_seqs[col_name]))

    # Build DataFrame from the intersection of all three
    common_ids = (
        set(all_seqs["cdna"].keys())
        & set(all_seqs["coding"].keys())
        & set(all_seqs["peptide"].keys())
    )
    if not common_ids:
        return pd.DataFrame(columns=["ensembl_transcript_id", "cdna", "coding", "peptide"])

    rows = []
    for tid in sorted(common_ids):
        rows.append({
            "ensembl_transcript_id": tid,
            "cdna": all_seqs["cdna"][tid],
            "coding": all_seqs["coding"][tid],
            "peptide": all_seqs["peptide"][tid],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def build_dataset(
    species: str,
    output_dir: Path,
    chromosomes: list[str] | None = None,
    force_rest_fallback: bool = False,
) -> Path:
    """Build the parquet dataset for a species."""
    config = SPECIES_CONFIG[species]
    dataset_name = config["dataset"]
    chroms = chromosomes or config["chromosomes"]

    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs: list[pd.DataFrame] = []
    biomart_failures: list[str] = []

    for chrom in chroms:
        if force_rest_fallback:
            df = fetch_sequences_rest_api(dataset_name, chrom)
        else:
            try:
                df = fetch_chromosome(dataset_name, chrom)
            except Exception:
                log.warning(
                    "BioMart failed for chr%s, falling back to REST API",
                    chrom,
                    exc_info=True,
                )
                biomart_failures.append(chrom)
                df = fetch_sequences_rest_api(dataset_name, chrom)

        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError(f"No data fetched for {species}")

    combined = pd.concat(all_dfs, ignore_index=True)

    # De-duplicate: keep first occurrence per transcript
    combined = combined.drop_duplicates(subset="ensembl_transcript_id", keep="first")

    # Compute CDS start
    log.info("Computing CDS start positions for %d transcripts...", len(combined))
    combined["cds_start_nt"] = combined.apply(
        lambda row: compute_cds_start(row["cdna"], row["coding"]),
        axis=1,
    )

    # Drop rows where CDS start couldn't be computed
    n_before = len(combined)
    combined = combined.dropna(subset=["cds_start_nt"])
    combined["cds_start_nt"] = combined["cds_start_nt"].astype(int)
    n_dropped_cds = n_before - len(combined)
    if n_dropped_cds:
        log.warning("Dropped %d rows where CDS start could not be determined", n_dropped_cds)

    # Build final output
    result = pd.DataFrame({
        "transcript_id": combined["ensembl_transcript_id"],
        "transcript_sequence": combined["cdna"],
        "protein_sequence": combined["peptide"],
        "cds_start_nt": combined["cds_start_nt"],
        "organism": species,
    })

    # ---- Validation ----
    log.info("Running validation checks...")

    # 1. ATG check
    atg_ok = result.apply(
        lambda row: row["transcript_sequence"][row["cds_start_nt"] - 1 : row["cds_start_nt"] + 2] == "ATG",
        axis=1,
    )
    n_atg_fail = (~atg_ok).sum()
    if n_atg_fail:
        log.warning("ATG check failed for %d / %d rows", n_atg_fail, len(result))
    else:
        log.info("ATG check passed for all %d rows", len(result))

    # 2. Protein starts with M
    m_ok = result["protein_sequence"].str.startswith("M")
    n_m_fail = (~m_ok).sum()
    if n_m_fail:
        log.warning("Protein M-start check failed for %d / %d rows", n_m_fail, len(result))
    else:
        log.info("Protein M-start check passed for all %d rows", len(result))

    # 3. No duplicate transcript IDs
    n_dup = result["transcript_id"].duplicated().sum()
    if n_dup:
        log.warning("Found %d duplicate transcript IDs", n_dup)
    else:
        log.info("No duplicate transcript IDs")

    # 4. Summary stats
    tx_lens = result["transcript_sequence"].str.len()
    cds_starts = result["cds_start_nt"]
    log.info(
        "Summary: %d rows | transcript len: median=%d, mean=%.0f | CDS start: median=%d",
        len(result),
        int(tx_lens.median()),
        tx_lens.mean(),
        int(cds_starts.median()),
    )

    # ---- Write parquet ----
    parquet_path = output_dir / f"ensembl_transcripts_{species}.parquet"
    result.to_parquet(parquet_path, index=False, engine="pyarrow")
    log.info("Wrote %s (%d rows)", parquet_path, len(result))

    # ---- Write metadata sidecar ----
    metadata = {
        "species": species,
        "dataset": dataset_name,
        "chromosomes_queried": chroms,
        "biomart_failures_fell_back_to_rest": biomart_failures,
        "force_rest_fallback": force_rest_fallback,
        "total_rows": len(result),
        "rows_dropped_no_cds_start": n_dropped_cds,
        "atg_check_failures": int(n_atg_fail),
        "protein_m_start_failures": int(n_m_fail),
        "duplicate_transcript_ids": int(n_dup),
        "median_transcript_length": int(tx_lens.median()),
        "mean_transcript_length": float(tx_lens.mean()),
        "median_cds_start": int(cds_starts.median()),
        "mean_cds_start": float(cds_starts.mean()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = output_dir / f"ensembl_transcripts_{species}_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Wrote %s", metadata_path)

    return parquet_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build transcript + protein dataset from Ensembl BioMart.",
    )
    parser.add_argument(
        "--species",
        type=str,
        default="homo_sapiens",
        choices=list(SPECIES_CONFIG.keys()),
        help="Species to query (default: homo_sapiens)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--chromosomes",
        nargs="+",
        type=str,
        default=None,
        help="Subset of chromosomes to query (default: all for species)",
    )
    parser.add_argument(
        "--force-rest-fallback",
        action="store_true",
        help="Skip pybiomart and use REST API for all chromosomes",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        species=args.species,
        output_dir=args.output_dir,
        chromosomes=args.chromosomes,
        force_rest_fallback=args.force_rest_fallback,
    )
