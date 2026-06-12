"""
Psychometric-style visualizations — Prior RDM study
=====================================================
Two figures (accuracy + RT), each with one panel per condition.
Replicates the Perugini et al. (2016) style:
  - X axis: signed coherence (right motion = positive, left motion = negative)
  - Y axis: P(respond right)  OR  median RT
  - Lines: HC (blue) vs PD (orange), mean ± SEM across subjects

Tech: plotnine
"""

import polars as pl
import pandas as pd
import numpy as np
import os
from plotnine import (
    ggplot, aes,
    geom_line, geom_point, geom_ribbon, geom_hline, geom_vline,
    facet_wrap, scale_color_manual, scale_fill_manual,
    scale_x_continuous, scale_y_continuous,
    labs, theme_bw, theme, element_text, element_line, element_blank,
    coord_cartesian,
)
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH  = "/sessions/amazing-affectionate-fermat/mnt/Prior_RDM_Analysis/data/behavior/processed/trials_primary.csv"
OUTPUT_DIR = "/sessions/amazing-affectionate-fermat/mnt/Prior_RDM_Analysis/data/behavior/processed/figures"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
GROUP_COLORS = {"HC": "#2166AC", "PD": "#D6604D"}
GROUP_FILL   = {"HC": "#6BAED6", "PD": "#FC8D59"}   # lighter fill for ribbon

CONDITION_ORDER  = ["Di_null", "Di_part", "Di_full"]
CONDITION_LABELS = {
    "Di_null": "Dichromatic\nnon-informative",
    "Di_part": "Dichromatic\npartially informative",
    "Di_full": "Dichromatic\nfully informative",
}

# X-axis tick positions and labels
COH_TICKS  = [-0.3333, -0.1333, -0.0667, 0.0, 0.0667, 0.1333, 0.3333]
COH_LABELS = ["-33 %", "-13 %", "-6.7 %", "0 %", "6.7 %", "13 %", "33 %"]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_and_prep(path: str) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=0, ignore_errors=True)
    df = df.with_columns([
        pl.col("RT").cast(pl.Float64, strict=False),
        pl.col("Correct").cast(pl.Int64, strict=False),
        pl.col("Coherence_total").cast(pl.Float64, strict=False),
    ])
    # Signed coherence: right = positive, left = negative
    df = df.with_columns(
        pl.when(pl.col("Direction") == "right")
            .then(pl.col("Coherence_total"))
            .otherwise(-pl.col("Coherence_total"))
            .alias("coh_signed")
    )
    # Binary: did participant respond right?
    df = df.with_columns(
        (pl.col("Response") == "right").cast(pl.Int64).alias("responded_right")
    )
    # Round signed coherence to avoid float noise
    df = df.with_columns(pl.col("coh_signed").round(4).alias("coh_signed"))
    return df


def subject_level_agg(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate to (subject × group × condition × coh_signed)."""
    return (
        df
        .group_by(["Part_Nr", "Group", "Condition", "coh_signed"])
        .agg([
            pl.col("responded_right").mean().alias("p_right"),
            pl.col("RT").median().alias("median_rt"),
            pl.len().alias("n_trials"),
        ])
        .sort(["Part_Nr", "Condition", "coh_signed"])
    )


def group_level_agg(subj: pl.DataFrame) -> pd.DataFrame:
    """Aggregate subject means to group mean ± SEM."""
    grp = (
        subj
        .group_by(["Group", "Condition", "coh_signed"])
        .agg([
            pl.col("p_right").mean().alias("mean_p_right"),
            pl.col("p_right").std().alias("sd_p_right"),
            pl.col("median_rt").mean().alias("mean_rt"),
            pl.col("median_rt").std().alias("sd_rt"),
            pl.len().alias("n_subjects"),
        ])
        .with_columns([
            (pl.col("sd_p_right") / pl.col("n_subjects").sqrt()).alias("se_p_right"),
            (pl.col("sd_rt")      / pl.col("n_subjects").sqrt()).alias("se_rt"),
        ])
        .sort(["Group", "Condition", "coh_signed"])
    )
    pdf = grp.to_pandas()

    # Ordered categorical for facet
    present = pdf["Condition"].unique().tolist()
    ordered = [c for c in CONDITION_ORDER if c in present]
    pdf["condition_label"] = pdf["Condition"].map(CONDITION_LABELS)
    cond_label_order = [CONDITION_LABELS[c] for c in ordered]
    pdf["condition_label"] = pd.Categorical(pdf["condition_label"],
                                             categories=cond_label_order, ordered=True)
    pdf["Group"] = pd.Categorical(pdf["Group"], categories=["HC", "PD"], ordered=True)
    return pdf


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_psychometric(grp: pd.DataFrame, output_dir: str) -> str:
    """P(respond right) vs signed coherence — one panel per condition."""
    p = (
        ggplot(grp, aes(x="coh_signed", y="mean_p_right",
                        color="Group", fill="Group", group="Group"))
        + geom_hline(yintercept=0.5, linetype="dashed", color="#AAAAAA", size=0.6)
        + geom_vline(xintercept=0.0, linetype="dashed", color="#AAAAAA", size=0.6)
        + geom_ribbon(
            aes(ymin="mean_p_right - se_p_right",
                ymax="mean_p_right + se_p_right"),
            alpha=0.20, color="none",
        )
        + geom_line(size=1.0)
        + geom_point(size=2.5)
        + facet_wrap("~ condition_label", ncol=1)
        + scale_color_manual(values=GROUP_COLORS)
        + scale_fill_manual(values=GROUP_FILL)
        + scale_x_continuous(
            breaks=COH_TICKS,
            labels=COH_LABELS,
            name="Signed coherence (right positive, left negative)",
        )
        + scale_y_continuous(
            breaks=[0.0, 0.25, 0.5, 0.75, 1.0],
            labels=["0 %", "25 %", "50 %", "75 %", "100 %"],
            name="P(respond right)",
        )
        + coord_cartesian(ylim=(0, 1))
        + labs(
            title="Psychometric functions: P(right response) by condition and group",
            color="Group", fill="Group",
        )
        + theme_bw()
        + theme(
            figure_size=(7, 10),
            strip_text=element_text(size=9),
            axis_text_x=element_text(size=8),
            axis_text_y=element_text(size=9),
            plot_title=element_text(size=11),
            legend_position="right",
            panel_spacing=0.4,
        )
    )
    out = os.path.join(output_dir, "psychometric_accuracy_by_condition.png")
    p.save(out, dpi=150, verbose=False)
    print(f"Saved: {out}")
    return out


def plot_rt_curve(grp: pd.DataFrame, output_dir: str) -> str:
    """Median RT vs signed coherence — one panel per condition."""
    p = (
        ggplot(grp, aes(x="coh_signed", y="mean_rt",
                        color="Group", fill="Group", group="Group"))
        + geom_vline(xintercept=0.0, linetype="dashed", color="#AAAAAA", size=0.6)
        + geom_ribbon(
            aes(ymin="mean_rt - se_rt", ymax="mean_rt + se_rt"),
            alpha=0.20, color="none",
        )
        + geom_line(size=1.0)
        + geom_point(size=2.5)
        + facet_wrap("~ condition_label", ncol=1)
        + scale_color_manual(values=GROUP_COLORS)
        + scale_fill_manual(values=GROUP_FILL)
        + scale_x_continuous(
            breaks=COH_TICKS,
            labels=COH_LABELS,
            name="Signed coherence (right positive, left negative)",
        )
        + scale_y_continuous(name="Median RT (s)")
        + labs(
            title="RT curves: median reaction time by condition and group",
            color="Group", fill="Group",
        )
        + theme_bw()
        + theme(
            figure_size=(7, 10),
            strip_text=element_text(size=9),
            axis_text_x=element_text(size=8),
            axis_text_y=element_text(size=9),
            plot_title=element_text(size=11),
            legend_position="right",
            panel_spacing=0.4,
        )
    )
    out = os.path.join(output_dir, "rt_curve_by_condition.png")
    p.save(out, dpi=150, verbose=False)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df   = load_and_prep(DATA_PATH)
    subj = subject_level_agg(df)
    grp  = group_level_agg(subj)

    print(f"Loaded {df.shape[0]} trials — {df['Part_Nr'].n_unique()} subjects")
    print(f"Conditions: {df['Condition'].unique().to_list()}")
    print(f"Group-level rows: {grp.shape[0]}")

    plot_psychometric(grp, OUTPUT_DIR)
    plot_rt_curve(grp, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
