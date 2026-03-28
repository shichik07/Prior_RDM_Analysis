import polars as pl
import os
import glob
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

def load_behavioral_data(data_path: str) -> pl.DataFrame:
    """
    Load all behavioral CSV files from the specified path and combine them into a single Polars DataFrame.
    """
    file_pattern = os.path.join(data_path, "RDM_*.csv")
    files = glob.glob(file_pattern)
    
    if not files:
        raise FileNotFoundError(f"No behavioral CSV files found in {data_path}")
    
    dfs = []
    for f in files:
        try:
            # Read all columns as string to avoid schema conflicts (e.g., Int64 vs String)
            # Use latin-1 encoding to handle non-UTF-8 characters (e.g., German umlauts)
            df = pl.read_csv(f, infer_schema_length=0, ignore_errors=True, encoding="latin-1")
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        raise ValueError("No data could be loaded from the CSV files.")
        
    return pl.concat(dfs, how="diagonal")

def clean_behavioral_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean the behavioral data:
    1. Cast necessary columns to correct types.
    2. Remove practice trials.
    3. Filter out trials with invalid RTs (e.g., < 150ms).
    4. Ensure correct data types.
    """
    # Cast necessary columns
    cleaned_df = df.with_columns([
        pl.col("RT").cast(pl.Float64, strict=False),
        pl.col("Correct").cast(pl.Int64, strict=False),
        pl.col("Trial_nr").cast(pl.Int64, strict=False),
        pl.col("Coherence_total").cast(pl.Float64, strict=False)
    ])

    # 1. Remove practice trials (Block contains 'Practice')
    cleaned_df = cleaned_df.filter(~pl.col("Block").str.contains("Practice"))
    
    # 2. Filter RT outliers (RT < 150ms as anticipatory)
    # RT is in seconds based on the head output (e.g., 0.255...)
    cleaned_df = cleaned_df.filter(pl.col("RT") >= 0.150)
    
    # 3. Handle missing values in critical columns
    cleaned_df = cleaned_df.drop_nulls(subset=["RT", "Correct"])
    
    return cleaned_df

def calculate_rt_distributions(df: pl.DataFrame, group_cols: List[str]) -> pl.DataFrame:
    """
    Calculate RT distribution statistics (mean, median, SD, and quantiles).
    """
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Define aggregation expressions
    agg_exprs = [
        pl.col("RT").mean().alias("rt_mean"),
        pl.col("RT").median().alias("rt_median"),
        pl.col("RT").std().alias("rt_std"),
        pl.col("RT").count().alias("trial_count")
    ]
    
    # Add quantiles
    for q in quantiles:
        agg_exprs.append(pl.col("RT").quantile(q).alias(f"rt_q{int(q*100)}"))
        
    return df.group_by(group_cols).agg(agg_exprs)

def calculate_error_rates(df: pl.DataFrame, group_cols: List[str]) -> pl.DataFrame:
    """
    Calculate error rates (1 - accuracy).
    """
    return df.group_by(group_cols).agg([
        (1 - pl.col("Correct").mean()).alias("error_rate"),
        pl.col("Correct").count().alias("trial_count")
    ])

def main(input_path: str, output_path: str):
    """
    Main pipeline execution.
    """
    print(f"Loading data from {input_path}...")
    df = load_behavioral_data(input_path)
    
    print("Cleaning data...")
    cleaned_df = clean_behavioral_data(df)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save cleaned trials
    cleaned_trials_path = os.path.join(output_path, "cleaned_trials.csv")
    cleaned_df.write_csv(cleaned_trials_path)
    print(f"Cleaned trials saved to {cleaned_trials_path}")
    
    # Calculate summaries
    # 1. Subject level
    subject_summary = calculate_rt_distributions(cleaned_df, ["Part_Nr", "Group"])
    subject_error = calculate_error_rates(cleaned_df, ["Part_Nr", "Group"])
    subject_stats = subject_summary.join(subject_error, on=["Part_Nr", "Group"], suffix="_err")
    
    # 2. Subject + Condition level
    condition_summary = calculate_rt_distributions(cleaned_df, ["Part_Nr", "Group", "Condition"])
    condition_error = calculate_error_rates(cleaned_df, ["Part_Nr", "Group", "Condition"])
    condition_stats = condition_summary.join(condition_error, on=["Part_Nr", "Group", "Condition"], suffix="_err")
    
    # 3. Subject + Condition + Coherence level
    coherence_summary = calculate_rt_distributions(cleaned_df, ["Part_Nr", "Group", "Condition", "Coherence_total"])
    coherence_error = calculate_error_rates(cleaned_df, ["Part_Nr", "Group", "Condition", "Coherence_total"])
    coherence_stats = coherence_summary.join(coherence_error, on=["Part_Nr", "Group", "Condition", "Coherence_total"], suffix="_err")
    
    # Save summaries
    subject_stats.write_csv(os.path.join(output_path, "subject_summary.csv"))
    condition_stats.write_csv(os.path.join(output_path, "condition_summary.csv"))
    coherence_stats.write_csv(os.path.join(output_path, "coherence_summary.csv"))
    
    print(f"Summaries saved to {output_path}")

if __name__ == "__main__":
    INPUT_DIR = "/mnt/e/priorRDM/Study/BehavioralData"
    OUTPUT_DIR = "/mnt/d/Data/Dropbox/PhD_Thesis/UniOL/Julius/Prior_RDM_Analysis/data/behavior/processed"
    main(INPUT_DIR, OUTPUT_DIR)
