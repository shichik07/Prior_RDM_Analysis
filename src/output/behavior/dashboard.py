import streamlit as st
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="Prior RDM Behavioral Dashboard", layout="wide")

@st.cache_data
def load_data():
    data_path = "/mnt/d/Data/Dropbox/PhD_Thesis/UniOL/Julius/Prior_RDM_Analysis/data/behavior/processed/cleaned_trials.csv"
    if not os.path.exists(data_path):
        st.error(f"Data file not found at {data_path}. Please run the preprocessing pipeline first.")
        return None
    
    # Load data with polars
    df = pl.read_csv(data_path)
    
    # Ensure types for filtering
    df = df.with_columns([
        pl.col("RT").cast(pl.Float64),
        pl.col("Correct").cast(pl.Int64),
        pl.col("Coherence_total").cast(pl.Float64),
        pl.col("Late_Response").cast(pl.String) # Cast to string for easier filtering if needed
    ])
    return df

def main():
    st.title("🧠 Prior RDM Behavioral Analysis Dashboard")
    st.markdown("Explore behavioral data from Parkinson's Disease (PD) and Healthy Control (HC) subjects.")

    df_raw = load_data()
    if df_raw is None:
        return

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filters")
    
    # Group Filter
    groups = df_raw["Group"].unique().to_list()
    selected_groups = st.sidebar.multiselect("Select Groups", options=groups, default=groups)
    
    # Condition Filter
    conditions = df_raw["Condition"].unique().to_list()
    selected_conditions = st.sidebar.multiselect("Select Conditions", options=conditions, default=conditions)
    
    # Coherence Filter
    coherences = sorted(df_raw["Coherence_total"].unique().to_list())
    selected_coherences = st.sidebar.multiselect("Select Coherence Levels", options=coherences, default=coherences)
    
    # Demographic Filters
    st.sidebar.subheader("Demographics & Flags")
    genders = df_raw["Gender"].unique().to_list()
    selected_genders = st.sidebar.multiselect("Gender", options=genders, default=genders)
    
    directions = df_raw["Direction"].unique().to_list()
    selected_directions = st.sidebar.multiselect("Direction", options=directions, default=directions)
    
    color_switches = df_raw["ColorSwitch"].unique().to_list()
    # Handle None/null in ColorSwitch
    color_switches = [str(x) if x is not None else "None" for x in color_switches]
    selected_switches = st.sidebar.multiselect("Color Switch", options=color_switches, default=color_switches)
    
    late_responses = df_raw["Late_Response"].unique().to_list()
    selected_late = st.sidebar.multiselect("Late Response", options=late_responses, default=late_responses)

    # --- APPLY FILTERS ---
    df = df_raw.filter(
        (pl.col("Group").is_in(selected_groups)) &
        (pl.col("Condition").is_in(selected_conditions)) &
        (pl.col("Coherence_total").is_in(selected_coherences)) &
        (pl.col("Gender").is_in(selected_genders)) &
        (pl.col("Direction").is_in(selected_directions)) &
        (pl.col("Late_Response").is_in(selected_late))
    )
    
    # Handle ColorSwitch separately due to nulls
    if "None" in selected_switches:
        df = df.filter(pl.col("ColorSwitch").is_null() | pl.col("ColorSwitch").cast(pl.String).is_in(selected_switches))
    else:
        df = df.filter(pl.col("ColorSwitch").cast(pl.String).is_in(selected_switches))

    if df.height == 0:
        st.warning("No data matches the selected filters.")
        return

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance Curves", 
        "📊 RT Distributions", 
        "📉 Quantile Analysis",
        "👤 Individual Subjects",
        "📄 Raw Data"
    ])

    # Convert filtered df to pandas for plotting
    pdf = df.to_pandas()
    pdf["accuracy"] = pdf["Correct"] # Use Correct as binary accuracy

    with tab1:
        st.header("Group Performance Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Psychometric Curves (Accuracy)")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.lineplot(data=pdf, x="Coherence_total", y="accuracy", hue="Group", style="Condition", marker="o", ax=ax, palette="viridis")
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("Coherence Level")
            st.pyplot(fig)
            
        with col2:
            st.subheader("Chronometric Curves (Mean RT)")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.lineplot(data=pdf, x="Coherence_total", y="RT", hue="Group", style="Condition", marker="s", ax=ax, palette="viridis")
            ax.set_ylabel("Mean RT (s)")
            ax.set_xlabel("Coherence Level")
            st.pyplot(fig)

    with tab2:
        st.header("Reaction Time Distributions")
        fig = sns.FacetGrid(pdf, row="Group", col="Condition", hue="Coherence_total", height=4, aspect=1.2, palette="viridis")
        fig.map(sns.kdeplot, "RT", fill=True, common_norm=False)
        fig.add_legend(title="Coherence")
        st.pyplot(fig.fig)

    with tab3:
        st.header("RT Quantile Analysis")
        st.info("Showing RT quantiles (10th, 30th, 50th, 70th, 90th) across groups and conditions.")
        
        # Calculate quantiles on the fly for the filtered data
        quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
        q_df = df.group_by(["Group", "Condition", "Coherence_total"]).agg([
            pl.col("RT").quantile(q).alias(f"q{int(q*100)}") for q in quantiles
        ]).to_pandas()
        
        q_melted = q_df.melt(id_vars=["Group", "Condition", "Coherence_total"], 
                             value_vars=[f"q{int(q*100)}" for q in quantiles],
                             var_name="Quantile", value_name="RT")
        q_melted["Quantile"] = q_melted["Quantile"].str.replace("q", "").astype(int)
        
        fig = sns.FacetGrid(q_melted, row="Group", col="Condition", hue="Coherence_total", height=4, aspect=1.2, palette="viridis")
        fig.map_dataframe(sns.lineplot, x="Quantile", y="RT", marker="o")
        fig.add_legend(title="Coherence")
        st.pyplot(fig.fig)

    with tab4:
        st.header("Individual Subject Exploration")
        subject = st.selectbox("Select Subject", options=sorted(pdf["Part_Nr"].unique()))
        
        sub_pdf = pdf[pdf["Part_Nr"] == subject]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"RT Distribution for {subject}")
            fig, ax = plt.subplots()
            sns.kdeplot(data=sub_pdf, x="RT", hue="Condition", fill=True, ax=ax)
            st.pyplot(fig)
            
        with col2:
            st.subheader(f"Accuracy per Condition for {subject}")
            acc_sub = sub_pdf.groupby("Condition")["accuracy"].mean().reset_index()
            fig, ax = plt.subplots()
            sns.barplot(data=acc_sub, x="Condition", y="accuracy", ax=ax, palette="muted")
            ax.set_ylim(0, 1.1)
            st.pyplot(fig)

    with tab5:
        st.header("Trial-Level Data")
        st.write(f"Showing {df.height} trials matching filters.")
        st.dataframe(pdf.head(1000))
        
        csv = pdf.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name='filtered_behavioral_data.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
