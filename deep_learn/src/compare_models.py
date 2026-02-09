"""
Cross-model comparison tool.

Scans reports/ for metadata.json files, builds a comparison table,
and generates a bar chart + table as PDF and CSV.

Usage:
    python compare_models.py                        # scan default reports dir
    python compare_models.py /path/to/reports/      # scan specific directory
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import REPORTS_DIR


def discover_reports(search_dir: str) -> list[dict]:
    """Find all metadata.json files under search_dir and parse them."""
    pattern = os.path.join(search_dir, "**", "metadata.json")
    reports = []
    for path in glob.glob(pattern, recursive=True):
        with open(path) as f:
            data = json.load(f)
            data["_json_path"] = path
            reports.append(data)
    return reports


def build_comparison_table(reports: list[dict]) -> pd.DataFrame:
    """Extract key metrics from each report into a DataFrame."""
    rows = []
    for r in reports:
        metrics = r.get("metrics", {})
        split = r.get("split_info", {})
        rows.append({
            "model_name": r.get("model_name", "unknown"),
            "timestamp": r.get("timestamp", ""),
            "accuracy": metrics.get("accuracy"),
            "cohen_kappa": metrics.get("cohen_kappa"),
            "f1_weighted": metrics.get("f1_weighted"),
            "f1_macro": metrics.get("f1_macro"),
            "eval_level": metrics.get("level", "test"),
            "test_count": split.get("test_count"),
            "split_method": split.get("split_method"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("f1_weighted", ascending=False).reset_index(drop=True)
    return df


def render_comparison_bar_chart(df: pd.DataFrame, pdf: PdfPages):
    """Grouped bar chart comparing models on key metrics."""
    metric_cols = ["accuracy", "cohen_kappa", "f1_weighted", "f1_macro"]
    available = [c for c in metric_cols if c in df.columns and df[c].notna().any()]
    if not available:
        return

    x = np.arange(len(df))
    width = 0.8 / len(available)

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 2), 6))
    for i, col in enumerate(available):
        label = col.replace("_", " ").title()
        ax.bar(x + i * width, df[col].fillna(0), width, label=label)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels(df["model_name"], rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def render_comparison_table(df: pd.DataFrame, pdf: PdfPages):
    """Render the comparison DataFrame as a matplotlib table."""
    display_cols = ["model_name", "accuracy", "cohen_kappa", "f1_weighted", "f1_macro"]
    display = df[[c for c in display_cols if c in df.columns]].copy()

    # Format numeric columns
    for col in display.columns:
        if col != "model_name":
            display[col] = display[col].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "N/A")

    fig, ax = plt.subplots(figsize=(10, max(3, len(display) * 0.5 + 1)))
    ax.axis("off")
    ax.set_title("Model Comparison Summary", fontsize=14, fontweight="bold", pad=20)

    col_labels = [c.replace("_", " ").title() for c in display.columns]
    table = ax.table(
        cellText=display.values,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(display) + 1):
        color = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare model reports")
    parser.add_argument(
        "dir",
        nargs="?",
        default=REPORTS_DIR,
        help="Directory to search for report metadata (default: config.REPORTS_DIR)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path prefix (without extension). Defaults to {dir}/model_comparison",
    )
    args = parser.parse_args()

    search_dir = args.dir
    output_prefix = args.output or os.path.join(search_dir, "model_comparison")

    reports = discover_reports(search_dir)
    if not reports:
        print(f"No metadata.json files found under {search_dir}")
        sys.exit(1)

    df = build_comparison_table(reports)

    print(f"\n=== Model Comparison ({len(df)} models) ===")
    print(df.to_string(index=False))

    # Save CSV
    csv_path = output_prefix + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Save PDF
    pdf_path = output_prefix + ".pdf"
    with PdfPages(pdf_path) as pdf:
        render_comparison_table(df, pdf)
        render_comparison_bar_chart(df, pdf)
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
