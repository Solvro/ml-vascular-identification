#!/usr/bin/env python3
"""
Generate publication-ready figures and tables for Open-Set Recognition paper.

This script loads all experiment results and creates:
- Comparison tables (LaTeX format)
- ROC/DET curves
- CMC curves
- Performance comparisons bar plots
"""

import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

warnings.filterwarnings("ignore")

# Setup plotting
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
SMALL_SIZE = 10
MEDIUM_SIZE = 12
LARGE_SIZE = 14
plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=LARGE_SIZE, labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)

PROJECT_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
TABLES_DIR = PROJECT_DIR / "tables"
FIGURES_DIR = TABLES_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

print("📊 Generating Paper Figures and Tables")
print(f"📂 Output directory: {FIGURES_DIR}")


def load_all_results() -> List[Dict]:
    """Load all experiment results from prototypes.pt files."""
    results = []

    for prototypes_file in OUTPUTS_DIR.rglob("prototypes.pt"):
        try:
            data = torch.load(prototypes_file, map_location="cpu")
            metrics = data.get("metrics", {})

            # Extract path info
            parts = prototypes_file.parts
            dataset = None
            model = None

            for part in parts:
                if part in ["mmcbnu", "dorsal", "utfvp", "fyo"]:
                    dataset = part
                if part in [
                    "basic",
                    "unet",
                    "attention_unet",
                    "resnet50_cbam_center",
                    "resnet50_cbam",
                    "simple_cnn",
                ]:
                    model = part

            if dataset and model:
                result = {
                    "dataset": dataset,
                    "model": model,
                    "auroc": float(metrics.get("auroc", np.nan)),
                    "oscr": float(metrics.get("oscr_auc", np.nan)),
                    "eer": float(metrics.get("eer", np.nan)),
                    "rank1": float(metrics.get("cmc_rank1", np.nan)),
                    "rank5": float(metrics.get("cmc_rank5", np.nan)),
                    "rank10": float(metrics.get("cmc_rank10", np.nan)),
                    "accuracy": float(metrics.get("accuracy_known", np.nan)),
                    "tpr_at_fpr_001": float(metrics.get("tpr_at_fpr_0.001", np.nan)),
                    "tpr_at_fpr_01": float(metrics.get("tpr_at_fpr_0.01", np.nan)),
                    "tpr_at_fpr_1": float(metrics.get("tpr_at_fpr_0.1", np.nan)),
                    "registration_time_ms": float(
                        metrics.get("registration_time_ms", np.nan)
                    ),
                    "query_time_ms": float(
                        metrics.get("avg_query_time_per_sample_ms", np.nan)
                    ),
                }
                results.append(result)
        except Exception as e:
            print(f"⚠️  Skipped {prototypes_file}: {e}")

    return results


def create_table2_architecture(df: pd.DataFrame) -> pd.DataFrame:
    """Table 2: Architecture Comparison on MMCBNU."""
    df_table = df[df["dataset"] == "mmcbnu"].copy()

    if df_table.empty:
        print("⚠️  No MMCBNU results found for Table 2")
        return pd.DataFrame()

    # Select and rename columns
    table = df_table[
        ["model", "oscr", "auroc", "eer", "rank1", "accuracy", "query_time_ms"]
    ].copy()
    table.columns = [
        "Model",
        "OSCR",
        "AUROC",
        "EER (%)",
        "Rank-1",
        "Accuracy",
        "Time (ms/sample)",
    ]

    # Format numbers
    for col in ["OSCR", "AUROC", "Rank-1", "Accuracy"]:
        table[col] = table[col].apply(
            lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A"
        )
    table["EER (%)"] = table["EER (%)"].apply(
        lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A"
    )
    table["Time (ms/sample)"] = table["Time (ms/sample)"].apply(
        lambda x: f"{x:.3f}" if not np.isnan(x) else "N/A"
    )

    # Save CSV
    csv_path = TABLES_DIR / "table2_architecture.csv"
    table.to_csv(csv_path, index=False)

    # Generate LaTeX
    latex = table.to_latex(index=False, escape=False)
    latex_path = TABLES_DIR / "table2_architecture.tex"
    with open(latex_path, "w") as f:
        f.write(latex)

    print("✅ Table 2 saved:")
    print(f"   CSV: {csv_path}")
    print(f"   LaTeX: {latex_path}")
    print(table.to_string())

    return table


def create_table3_generalization(df: pd.DataFrame) -> pd.DataFrame:
    """Table 3: Generalization across datasets."""
    df_table = df[df["model"] == "resnet50_cbam_center"].copy()

    if df_table.empty:
        print("⚠️  No generalization results found for Table 3")
        return pd.DataFrame()

    table = df_table[["dataset", "oscr", "auroc", "eer", "rank1", "accuracy"]].copy()
    table.columns = ["Dataset", "OSCR", "AUROC", "EER (%)", "Rank-1", "Accuracy"]

    # Format numbers
    for col in ["OSCR", "AUROC", "Rank-1", "Accuracy"]:
        table[col] = table[col].apply(
            lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A"
        )
    table["EER (%)"] = table["EER (%)"].apply(
        lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A"
    )

    # Save CSV
    csv_path = TABLES_DIR / "table3_generalization.csv"
    table.to_csv(csv_path, index=False)

    # Generate LaTeX
    latex = table.to_latex(index=False, escape=False)
    latex_path = TABLES_DIR / "table3_generalization.tex"
    with open(latex_path, "w") as f:
        f.write(latex)

    print("✅ Table 3 saved:")
    print(f"   CSV: {csv_path}")
    print(f"   LaTeX: {latex_path}")
    print(table.to_string())

    return table


def create_figure_arch_comparison(df: pd.DataFrame):
    """Figure: Architecture comparison bar plot."""
    df_fig = df[df["dataset"] == "mmcbnu"].copy()

    if df_fig.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Architecture Comparison on MMCBNU", fontsize=16, fontweight="bold")

    metrics = ["oscr", "auroc", "rank1", "accuracy"]
    titles = ["OSCR", "AUROC", "Rank-1 Accuracy", "Overall Accuracy"]

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        data = df_fig[["model", metric]].dropna()

        bars = ax.bar(
            range(len(data)),
            data[metric].values,
            color=sns.color_palette("husl", len(data)),
        )
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data["model"].values, rotation=45, ha="right")
        ax.set_ylabel(title, fontweight="bold")
        ax.set_ylim([0, 1])
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    fig_path = FIGURES_DIR / "figure_architecture_comparison.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure saved: {fig_path}")


def create_figure_generalization(df: pd.DataFrame):
    """Figure: Generalization across datasets."""
    df_fig = df[df["model"] == "resnet50_cbam_center"].copy()

    if df_fig.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Generalization: Best Model on Different Datasets",
        fontsize=16,
        fontweight="bold",
    )

    # OSCR comparison
    ax = axes[0]
    data = df_fig[["dataset", "oscr"]].dropna()
    bars = ax.bar(
        range(len(data)),
        data["oscr"].values,
        color=sns.color_palette("Set2", len(data)),
    )
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data["dataset"].values)
    ax.set_ylabel("OSCR", fontweight="bold")
    ax.set_ylim([0, 1])
    ax.set_title("Open-Set Classification Rate")
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # AUROC comparison
    ax = axes[1]
    data = df_fig[["dataset", "auroc"]].dropna()
    bars = ax.bar(
        range(len(data)),
        data["auroc"].values,
        color=sns.color_palette("Set2", len(data)),
    )
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data["dataset"].values)
    ax.set_ylabel("AUROC", fontweight="bold")
    ax.set_ylim([0, 1])
    ax.set_title("Known vs Unknown Discrimination")
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    fig_path = FIGURES_DIR / "figure_generalization.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure saved: {fig_path}")


def create_cmc_curves(df: pd.DataFrame):
    """Figure: CMC curves across datasets."""
    df_fig = df[df["model"] == "resnet50_cbam_center"].dropna(
        subset=["rank1", "rank5", "rank10"]
    )

    if df_fig.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in df_fig.iterrows():
        ranks = np.array([row["rank1"], row["rank5"], row["rank10"]])
        rank_labels = np.array([1, 5, 10])
        ax.plot(
            rank_labels,
            ranks,
            marker="o",
            linewidth=2,
            markersize=8,
            label=row["dataset"].upper(),
        )

    ax.set_xlabel("Rank", fontweight="bold", fontsize=12)
    ax.set_ylabel("Cumulative Recognition Rate", fontweight="bold", fontsize=12)
    ax.set_title("CMC Curves - Generalization", fontsize=14, fontweight="bold")
    ax.set_xticks([1, 5, 10])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "figure_cmc_curves.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure saved: {fig_path}")


def main():
    """Generate all paper figures and tables."""
    print("🔍 Loading experiment results...")
    results = load_all_results()

    if not results:
        print("❌ No results found!")
        return

    df = pd.DataFrame(results)
    print(f"✅ Loaded {len(df)} experiments")
    print(f"   Datasets: {df['dataset'].unique()}")
    print(f"   Models: {df['model'].unique()}")
    print()

    # Generate tables
    print("📊 Generating Tables...")
    create_table2_architecture(df)
    print()
    create_table3_generalization(df)
    print()

    # Generate figures
    print("📈 Generating Figures...")
    create_figure_arch_comparison(df)
    create_figure_generalization(df)
    create_cmc_curves(df)

    print()
    print("════════════════════════════════════════════════════════════════")
    print("✅ All figures and tables generated successfully!")
    print(f"📂 Output directory: {FIGURES_DIR}")
    print("════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
