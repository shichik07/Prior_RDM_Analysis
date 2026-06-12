"""
Behavioral visualizations — Prior RDM study
============================================
Produces two faceted figures:
  1. Reaction time (median RT per subject) — rows: condition, cols: coherence, color: group
  2. Error rate — same facet structure

Tech: plotnine (ggplot2 for Python)
"""

import polars as pl
import pandas as pd
import os
from plotnine import (
    ggplot, aes,
    geom_boxplot, geom_jitter, geom_bar, geom_errorbar,
    facet_grid, scale_fill_manual, scale_color_manual,
    labs, theme_bw, theme, element_text,
    coord_cartesian, position_dodge,
)
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH  = "/sessions/amazing-affectionate-fermat/mnt/Prior_RDM_Analysis/data/behavior/processed/trials_primary.csv"
OUTPUT_DIR = "/sessions/amazing-affectionate-fermat/mnt/Prior_RDM_Analysis/data/behavior/processed/figures"

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
GROUP_COLORS   = {"HC": "#2166AC", "PD": "#D6604D"}
COHERENCE_ORDER = ["0 %", "6.7 %", "13.3 %", "33.3 %"]
CONDITION_ORDER = ["Monochromatic", "Di null", "Di part", "Di full"]


def load_data(path: str) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=0, ignore_errors=True)
    return df.with_columns([
        pl.col("RT").cast(pl.Float64, strict=False),
        pl.col("Correct").cast(pl.Int64, strict=False),
        pl.col("Coherence_total").cast(pl.Float64, strict=False),
    ])


def add_labels(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.when(pl.col("Coherence_total") < 0.001)
            .then(pl.lit("0 %"))
        .when(pl.col("Coherence_total") < 0.09)
            .then(pl.lit("6.7 %"))
        .when(pl.col("Coherence_total") < 0.20)
            .then(pl.lit("13.3 %"))
        .otherwise(pl.lit("33.3 %"))
        .alias("coherence_label")
    )
    condition_map = {"Mono": "Monochromatic", "Di_null": "Di null",
                     "Di_part": "Di part", "Di_full": "Di full"}
    return df.with_columns(
        pl.col("Condition").replace(condition_map).alias("condition_label")
    )


def make_subject_summary(df: pl.DataFrame) -> pd.DataFrame:
    agg = (
        df
        .group_by(["Part_Nr", "Group", "condition_label", "coherence_label"])
        .agg([
            pl.col("RT").median().alias("median_rt"),
            pl.col("Correct").mean().alias("accuracy"),
            pl.len().alias("n_trials"),
        ])
        .with_columns((1 - pl.col("accuracy")).alias("error_rate"))
    )

    pdf = agg.to_pandas()
    present = pdf["condition_label"].unique().tolist()
    ordered_cond  = [c for c in CONDITION_ORDER if c in present]
    pdf["condition_label"] = pd.Categorical(pdf["condition_label"], categories=ordered_cond, ordered=True)
    pdf["coherence_label"] = pd.Categorical(pdf["coherence_label"], categories=COHERENCE_ORDER, ordered=True)
    pdf["Group"]            = pd.Categorical(pdf["Group"], categories=["HC", "PD"], ordered=True)
    return pdf


def plot_rt(pdf: pd.DataFrame, output_dir: str) -> str:
    p = (
        ggplot(pdf, aes(x="Group", y="median_rt", fill="Group", color="Group"))
        + geom_boxplot(alpha=0.35, outlier_shape="", width=0.55)
        + geom_jitter(width=0.12, height=0, size=1.8, alpha=0.7)
        + facet_grid("condition_label ~ coherence_label")
        + scale_fill_manual(values=GROUP_COLORS)
        + scale_color_manual(values=GROUP_COLORS)
        + labs(
            title="Reaction Time by Condition, Coherence, and Group",
            x="Group", y="Median RT (s)", fill="Group", color="Group",
        )
        + theme_bw()
        + theme(
            figure_size=(12, 8),
            strip_text=element_text(size=9),
            axis_text_x=element_text(size=9),
            plot_title=element_text(size=12),
        )
    )
    out = os.path.join(output_dir, "rt_by_condition_coherence_group.png")
    p.save(out, dpi=150, verbose=False)
    print(f"Saved: {out}")
    return out


def plot_error_rate(pdf: pd.DataFrame, output_dir: str) -> str:
    # Group-level summary for bars + SE
    grp = (
        pdf
        .groupby(["Group", "condition_label", "coherence_label"], observed=True)
        .agg(
            mean_error=("error_rate", "mean"),
            se_error=("error_rate", lambda x: x.std(ddof=1) / (len(x) ** 0.5)),
        )
        .reset_index()
    )

    p = (
        ggplot(grp, aes(x="Group", y="mean_error", fill="Group", color="Group"))
        + geom_bar(stat="identity", width=0.55, alpha=0.40)
        + geom_errorbar(
            aes(ymin="mean_error - se_error", ymax="mean_error + se_error"),
            width=0.2, size=0.7,
        )
        + geom_jitter(
            data=pdf,
            mapping=aes(x="Group", y="error_rate", color="Group"),
            width=0.12, height=0, size=1.8, alpha=0.6, inherit_aes=False,
        )
        + facet_grid("condition_label ~ coherence_label")
        + scale_fill_manual(values=GROUP_COLORS)
        + scale_color_manual(values=GROUP_COLORS)
        + coord_cartesian(ylim=(0, 1))
        + labs(
            title="Error Rate by Condition, Coherence, and Group",
            x="Group", y="Error rate", fill="Group", color="Group",
        )
        + theme_bw()
        + theme(
            figure_size=(12, 8),
            strip_text=element_text(size=9),
            axis_text_x=element_text(size=9),
            plot_title=element_text(size=12),
        )
    )
    out = os.path.join(output_dir, "error_rate_by_condition_coherence_group.png")
    p.save(out, dpi=150, verbose=False)
    print(f"Saved: {out}")
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    df = add_labels(df)
    print(f"Loaded {df.shape[0]} trials — {df['Part_Nr'].n_unique()} subjects")
    pdf = make_subject_summary(df)
    plot_rt(pdf, OUTPUT_DIR)
    plot_error_rate(pdf, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
