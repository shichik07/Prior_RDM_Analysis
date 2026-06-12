"""
Behavioral preprocessing pipeline — Prior RDM study
=====================================================
Applies RT exclusion criteria and produces analysis-ready datasets.

RT cutoffs (from study_parameters.md / PREP-05 ticket):
  - Lower:  RT < 0.200 s  → excluded (anticipatory)
  - Nominal upper: RT >= 2.500 s → excluded from primary analysis
  - Flag zone: 2.500 s <= RT <= 3.000 s → kept in flagged dataset, exploratory only
  - Hard ceiling: RT > 3.000 s → excluded from everything

Tech stack: Polars
"""

import polars as pl
import os
import glob
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DATA_PATH   = "/mnt/e/priorRDM/Study/BehavioralData"
INTERIM_CSV     = "/sessions/amazing-affectionate-fermat/mnt/Prior_RDM_Analysis/data/behavior/processed/cleaned_trials.csv"
OUTPUT_DIR      = "/sessions/amazing-affectionate-fermat/mnt/Prior_RDM_Analysis/data/behavior/processed"

# ---------------------------------------------------------------------------
# RT cutoffs
# ---------------------------------------------------------------------------
RT_LOWER        = 0.200   # seconds — anticipatory responses
RT_UPPER        = 2.500   # seconds — nominal ceiling (primary analysis)
RT_FLAG_UPPER   = 3.000   # seconds — hard ceiling; 2.5–3.0 is the "flag zone"

# ---------------------------------------------------------------------------
# Condition label order for downstream use
# ---------------------------------------------------------------------------
CONDITION_ORDER = ["Mono", "Di_null", "Di_part", "Di_full"]
COHERENCE_LABELS = {
    0.0:    "0 %",
    0.0667: "6.7 %",
    0.1333: "13.3 %",
    0.3333: "33.3 %",
}


def load_raw(path: str) -> pl.DataFrame:
    """Load all RDM_*.csv files from raw data directory."""
    pattern = os.path.join(path, "RDM_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No RDM_*.csv files found in {path}")
    dfs = []
    for f in files:
        try:
            df = pl.read_csv(f, infer_schema_length=0, ignore_errors=True, encoding="latin-1")
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")
    return pl.concat(dfs, how="diagonal")


def load_interim(path: str) -> pl.DataFrame:
    """Load the already-concatenated interim CSV (fallback when raw drive unavailable)."""
    print(f"Loading interim CSV: {path}")
    return pl.read_csv(path, infer_schema_length=0, ignore_errors=True)


def cast_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Cast key columns to correct types."""
    return df.with_columns([
        pl.col("RT").cast(pl.Float64, strict=False),
        pl.col("Correct").cast(pl.Int64, strict=False),
        pl.col("Trial_nr").cast(pl.Int64, strict=False),
        pl.col("Coherence_total").cast(pl.Float64, strict=False),
    ])


def remove_practice(df: pl.DataFrame) -> pl.DataFrame:
    """Drop practice trials (Block contains 'Practice')."""
    return df.filter(~pl.col("Block").str.contains("Practice"))


def apply_rt_cutoffs(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Apply RT exclusions and produce three datasets:

      primary   — trials used in all main analyses
                  RT >= RT_LOWER AND RT < RT_UPPER AND Correct is not null
      flagged   — primary + the 2.5–3.0 s zone (for exploratory analysis)
      excluded  — rows that were dropped, with reason column
    """
    # Drop rows with no RT or no Correct
    valid = df.drop_nulls(subset=["RT", "Correct"])

    # Annotate each trial with its RT status
    valid = valid.with_columns(
        pl.when(pl.col("RT") < RT_LOWER)
            .then(pl.lit("anticipatory"))
        .when(pl.col("RT") > RT_FLAG_UPPER)
            .then(pl.lit("too_slow"))
        .when(pl.col("RT") >= RT_UPPER)
            .then(pl.lit("flag_zone"))
        .otherwise(pl.lit("ok"))
        .alias("rt_status")
    )

    primary = valid.filter(pl.col("rt_status") == "ok")
    flagged = valid.filter(pl.col("rt_status").is_in(["ok", "flag_zone"]))
    excluded = valid.filter(pl.col("rt_status").is_in(["anticipatory", "too_slow"]))

    return {"primary": primary, "flagged": flagged, "excluded": excluded}


def round_coherence(df: pl.DataFrame) -> pl.DataFrame:
    """Round Coherence_total to 4 decimal places to avoid float noise."""
    return df.with_columns(
        pl.col("Coherence_total").round(4).alias("Coherence_total")
    )


def add_coherence_label(df: pl.DataFrame) -> pl.DataFrame:
    """Add a readable coherence label column."""
    return df.with_columns(
        pl.when(pl.col("Coherence_total") < 0.001)
            .then(pl.lit("0 %"))
        .when(pl.col("Coherence_total") < 0.09)
            .then(pl.lit("6.7 %"))
        .when(pl.col("Coherence_total") < 0.20)
            .then(pl.lit("13.3 %"))
        .otherwise(pl.lit("33.3 %"))
        .alias("coherence_label")
    )


def summarise_exclusions(original: pl.DataFrame, datasets: dict) -> pl.DataFrame:
    """Print and return a simple exclusion summary."""
    n_orig = original.shape[0]
    n_primary = datasets["primary"].shape[0]
    n_flag = datasets["flagged"].shape[0] - n_primary
    n_excluded = datasets["excluded"].shape[0]
    n_null = original.shape[0] - original.drop_nulls(subset=["RT", "Correct"]).shape[0]

    summary = pl.DataFrame({
        "reason": ["missing RT/Correct", "anticipatory (RT < 0.2 s)",
                   "too slow (RT > 3.0 s)", "flag zone (2.5–3.0 s, in exploratory only)",
                   "retained (primary)"],
        "n_trials": [n_null,
                     datasets["excluded"].filter(pl.col("rt_status") == "anticipatory").shape[0],
                     datasets["excluded"].filter(pl.col("rt_status") == "too_slow").shape[0],
                     n_flag,
                     n_primary],
    })
    print("\n=== RT Exclusion Summary ===")
    print(summary)
    print(f"  Original rows:  {n_orig}")
    print(f"  Primary dataset: {n_primary} ({100*n_primary/n_orig:.1f} %)")
    return summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load ---
    try:
        raw = load_raw(RAW_DATA_PATH)
        print(f"Loaded {raw.shape[0]} rows from raw files")
        raw = remove_practice(raw)
    except FileNotFoundError:
        print("Raw data drive not mounted — falling back to interim CSV")
        raw = load_interim(INTERIM_CSV)

    # --- Cast & clean ---
    df = cast_columns(raw)
    df = round_coherence(df)
    df = add_coherence_label(df)

    # Drop explicit practice rows if Block column present
    if "Block" in df.columns:
        df = remove_practice(df)

    # --- RT cutoffs ---
    datasets = apply_rt_cutoffs(df)
    excl_summary = summarise_exclusions(df, datasets)

    # --- Save ---
    primary = datasets["primary"]
    flagged = datasets["flagged"]

    primary.write_csv(os.path.join(OUTPUT_DIR, "trials_primary.csv"))
    flagged.write_csv(os.path.join(OUTPUT_DIR, "trials_flagged.csv"))
    excl_summary.write_csv(os.path.join(OUTPUT_DIR, "exclusion_summary.csv"))

    print(f"\nSaved trials_primary.csv  ({primary.shape[0]} rows)")
    print(f"Saved trials_flagged.csv  ({flagged.shape[0]} rows)")
    print(f"Saved exclusion_summary.csv")

    # --- Quick subject-level counts ---
    subj_counts = (
        primary
        .group_by(["Part_Nr", "Group", "Condition"])
        .agg(pl.len().alias("n_trials"))
        .sort(["Group", "Part_Nr", "Condition"])
    )
    subj_counts.write_csv(os.path.join(OUTPUT_DIR, "trial_counts_by_subject_condition.csv"))
    print(f"Saved trial_counts_by_subject_condition.csv")

    # Report missing conditions
    found_conditions = primary["Condition"].unique().to_list()
    missing = [c for c in CONDITION_ORDER if c not in found_conditions]
    if missing:
        print(f"\n⚠  Missing conditions (raw data not loaded?): {missing}")

    return primary, flagged


if __name__ == "__main__":
    main()
