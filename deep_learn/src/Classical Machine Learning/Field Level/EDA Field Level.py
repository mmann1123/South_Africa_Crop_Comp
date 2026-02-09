"""
Field‑level EDA
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import WORKING_DIR

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

FIELD_PARQUET = os.path.join(WORKING_DIR, "Data", "field_data.parquet")

def main():
    df = pd.read_parquet(FIELD_PARQUET, engine="pyarrow")
    print("Loaded field_data:", df.shape)

    band_cols   = [c for c in df.columns if c.endswith("_median")]
    band_labels = [c.rsplit("_median", 1)[0] for c in band_cols]

    crop_order = df['crop_name'].value_counts().index.tolist()

    sns.set_theme(style="whitegrid", context="paper")
    palette = sns.color_palette("colorblind")

    # --- Plot 1: Number of fields per crop type ---
    plt.figure(figsize=(8,5))
    ax1 = sns.countplot(
        data=df,
        x='crop_name',
        order=crop_order,
        palette=palette
    )
    ax1.set_title("Number of Fields per Crop Type", fontsize=14)
    ax1.set_xlabel("Crop Type", fontsize=12)
    ax1.set_ylabel("Field Count", fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Average Field Area by crop type (descending) ---
    area_df = (
        df.groupby('crop_name')['SHAPE_AREA']
          .mean()
          .reset_index()
          .sort_values('SHAPE_AREA', ascending=False)
    )
    plt.figure(figsize=(8,5))
    ax2 = sns.barplot(
        data=area_df,
        x='crop_name',
        y='SHAPE_AREA',
        palette=palette
    )
    ax2.set_title("Average Field Area by Crop Type", fontsize=14)
    ax2.set_xlabel("Crop Type", fontsize=12)
    ax2.set_ylabel("Field Area", fontsize=12)  # renamed y-axis
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    # --- Plot 3: Clustered bar chart of mean band values per spectral band ---
    mean_bands = df.groupby('crop_name')[band_cols].mean().loc[crop_order]
    tb = mean_bands.T
    tb.index = band_labels  

    melted = (
        tb.reset_index()
          .melt(id_vars='index', var_name='Crop Type', value_name='Mean Value')
          .rename(columns={'index': 'Spectral Band'})
    )

    plt.figure(figsize=(12,6))
    ax3 = sns.barplot(
        data=melted,
        x='Spectral Band',
        y='Mean Value',
        hue='Crop Type',
        palette=palette
    )
    ax3.set_title("Mean Spectral Band Values by Crop Type", fontsize=14)
    ax3.set_xlabel("Spectral Band", fontsize=12)
    ax3.set_ylabel("Mean Band Value", fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(title="Crop Type", bbox_to_anchor=(1.02,1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
