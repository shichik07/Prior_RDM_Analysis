import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def load_data(data_dir: str):
    """Load the processed behavioral data."""
    cleaned_trials = pl.read_csv(os.path.join(data_dir, "cleaned_trials.csv"))
    subject_summary = pl.read_csv(os.path.join(data_dir, "subject_summary.csv"))
    condition_summary = pl.read_csv(os.path.join(data_dir, "condition_summary.csv"))
    coherence_summary = pl.read_csv(os.path.join(data_dir, "coherence_summary.csv"))
    return cleaned_trials, subject_summary, condition_summary, coherence_summary

def plot_rt_distributions(cleaned_trials: pl.DataFrame, output_dir: str):
    """Plot RT distributions faceted by Group and Condition, colored by Coherence."""
    df = cleaned_trials.to_pandas()
    
    # RT distributions faceted by Group and Condition, colored by Coherence_total
    # We might want to bin coherence if there are too many levels, 
    # but based on previous inspection there are only a few.
    g = sns.FacetGrid(df, row="Group", col="Condition", hue="Coherence_total", height=4, aspect=1.2, palette="viridis")
    g.map(sns.kdeplot, "RT", fill=True, common_norm=False)
    g.add_legend(title="Coherence")
    g.set_axis_labels("Reaction Time (s)", "Density")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("RT Distributions by Group, Condition, and Coherence")
    plt.savefig(os.path.join(output_dir, "rt_distribution_faceted.png"))
    plt.close()

def plot_performance_metrics(condition_summary: pl.DataFrame, output_dir: str):
    """Plot mean RT and error rates across conditions."""
    df = condition_summary.to_pandas()
    
    # Mean RT
    plt.figure(figsize=(10, 6))
    sns.pointplot(data=df, x="Condition", y="rt_mean", hue="Group", dodge=True, markers=["o", "s"], capsize=.1, palette="viridis")
    plt.title("Mean Reaction Time by Condition and Group")
    plt.ylabel("Mean RT (s)")
    plt.savefig(os.path.join(output_dir, "mean_rt_by_condition.png"))
    plt.close()

    # Error Rate
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Condition", y="error_rate", hue="Group", palette="viridis")
    plt.title("Error Rate by Condition and Group")
    plt.ylabel("Error Rate")
    plt.savefig(os.path.join(output_dir, "error_rate_by_condition.png"))
    plt.close()

def plot_rt_quantiles(condition_summary: pl.DataFrame, output_dir: str):
    """Plot RT quantiles faceted by Group and Condition."""
    # This function uses condition_summary which doesn't have coherence levels.
    # To see coherence effects on quantiles, we should use coherence_summary if we added quantiles there.
    # Wait, did I add quantiles to coherence_summary? 
    # Let's check preprocess.py. Yes, calculate_rt_distributions adds quantiles.
    pass

def plot_rt_quantiles_faceted(coherence_summary: pl.DataFrame, output_dir: str):
    """Plot RT quantiles faceted by Group and Condition, showing Coherence effects."""
    df = coherence_summary.to_pandas()
    
    quantile_cols = ["rt_q10", "rt_q30", "rt_q50", "rt_q70", "rt_q90"]
    df_melted = df.melt(id_vars=["Part_Nr", "Group", "Condition", "Coherence_total"], 
                        value_vars=quantile_cols, var_name="Quantile", value_name="RT")
    df_melted["Quantile"] = df_melted["Quantile"].str.replace("rt_q", "").astype(int)

    g = sns.FacetGrid(df_melted, row="Group", col="Condition", hue="Coherence_total", height=4, aspect=1.2, palette="viridis")
    g.map_dataframe(sns.lineplot, x="Quantile", y="RT", markers=True)
    g.add_legend(title="Coherence")
    g.set_axis_labels("Quantile (%)", "Reaction Time (s)")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("RT Quantiles by Group, Condition, and Coherence")
    plt.savefig(os.path.join(output_dir, "rt_quantiles_faceted.png"))
    plt.close()

def plot_coherence_effects(coherence_summary: pl.DataFrame, output_dir: str):
    """Plot RT and error rates as a function of coherence, overlaying Groups and faceting by Condition."""
    df = coherence_summary.to_pandas()
    df["accuracy"] = 1 - df["error_rate"]
    
    # Psychometric Curve: Group overlay, Condition facets
    g = sns.FacetGrid(df, col="Condition", height=5, aspect=1.2)
    g.map_dataframe(sns.lineplot, x="Coherence_total", y="accuracy", hue="Group", marker="o", palette="viridis")
    g.add_legend(title="Group")
    g.set_axis_labels("Coherence Level", "Accuracy")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Psychometric Curves: HC vs PD by Condition")
    plt.savefig(os.path.join(output_dir, "psychometric_curves_overlay.png"))
    plt.close()

    # Chronometric Curve: Group overlay, Condition facets
    g = sns.FacetGrid(df, col="Condition", height=5, aspect=1.2)
    g.map_dataframe(sns.lineplot, x="Coherence_total", y="rt_mean", hue="Group", marker="s", palette="viridis")
    g.add_legend(title="Group")
    g.set_axis_labels("Coherence Level", "Mean RT (s)")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Chronometric Curves: HC vs PD by Condition")
    plt.savefig(os.path.join(output_dir, "chronometric_curves_overlay.png"))
    plt.close()

def plot_subject_variability(cleaned_trials: pl.DataFrame, output_dir: str):
    """Plot individual RT distributions."""
    df = cleaned_trials.to_pandas()
    subjects = df["Part_Nr"].unique()
    
    # Limit to first 12 subjects for the grid plot to keep it legible
    sample_subjects = subjects[:12]
    df_sample = df[df["Part_Nr"].isin(sample_subjects)]
    
    g = sns.FacetGrid(df_sample, col="Part_Nr", hue="Group", col_wrap=4, height=3, sharex=True, sharey=False, palette="viridis")
    g.map(sns.kdeplot, "RT", fill=True)
    g.set_axis_labels("RT (s)", "Density")
    g.set_titles("{col_name}")
    g.add_legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "subject_rt_distributions_sample.png"))
    plt.close()

def main():
    data_dir = "/mnt/d/Data/Dropbox/PhD_Thesis/UniOL/Julius/Prior_RDM_Analysis/data/behavior/processed"
    output_dir = os.path.join(data_dir, "figures")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from {data_dir}...")
    cleaned_trials, subject_summary, condition_summary, coherence_summary = load_data(data_dir)

    print("Generating RT distribution plots...")
    plot_rt_distributions(cleaned_trials, output_dir)

    print("Generating performance metric plots...")
    plot_performance_metrics(condition_summary, output_dir)

    print("Generating quantile plots...")
    plot_rt_quantiles_faceted(coherence_summary, output_dir)

    print("Generating coherence effect plots...")
    plot_coherence_effects(coherence_summary, output_dir)

    print("Generating subject variability plots...")
    plot_subject_variability(cleaned_trials, output_dir)

    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
