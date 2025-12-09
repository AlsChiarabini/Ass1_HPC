import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================
# CONFIGURAZIONE
# ============================

CSV_FILE = "risultati_atax.csv"   # il tuo CSV
PLOTS_DIR = Path("plots")

DATASETS = ["MINI", "SMALL", "STANDARD", "LARGE", "EXTRALARGE"]

# Baseline speedup: scegli qualsiasi kernel
BASELINE_KERNEL = "atax_base_const.cu"

sns.set_theme(style="whitegrid", palette="husl")


# ============================
# UTILITIES
# ============================

def ensure_plots_dir():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(filepath):
    df = pd.read_csv(filepath)

    # Verifica colonne
    assert "file" in df.columns
    assert "dataset" in df.columns
    assert "avg_time_ms" in df.columns

    return df


# ============================
# PLOT: Execution Time per Dataset
# ============================

def plot_times_per_dataset(df, dataset, out_dir):
    data = df[df["dataset"] == dataset].copy()

    if data.empty:
        print(f"[WARN] Dataset {dataset} non trovato.")
        return

    data = data.sort_values("avg_time_ms")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=data, x="file", y="avg_time_ms", hue="file",
                ax=ax, palette="husl", legend=False)

    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title(f"ATAX — Execution Time — {dataset}")

    # etichette sopra le barre
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset.lower()}_time.png", dpi=150)
    plt.close()


# ============================
# PLOT: Speedup per Dataset
# ============================

def plot_speedup(df, dataset, out_dir):
    data = df[df["dataset"] == dataset].copy()

    if data.empty:
        print(f"[WARN] Dataset {dataset} non trovato.")
        return

    baseline = data[data["file"] == BASELINE_KERNEL]
    if baseline.empty:
        print(f"[ERROR] Baseline {BASELINE_KERNEL} non presente nel dataset {dataset}.")
        return

    baseline_time = baseline["avg_time_ms"].values[0]

    data["speedup"] = baseline_time / data["avg_time_ms"]
    data = data.sort_values("speedup")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=data, x="file", y="speedup",
                 marker='o', linewidth=2.5, ax=ax, color="#2ecc71")

    plt.xticks(rotation=45, ha="right")
    ax.set_ylabel("Speedup (vs baseline)")
    ax.set_title(f"ATAX — Speedup — {dataset}")

    ax.axhline(1.0, color="red", linestyle="--", linewidth=2, alpha=0.7)

    # Etichette
    for x, y in enumerate(data["speedup"]):
        ax.text(x, y + 0.05, f"{y:.2f}×",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset.lower()}_speedup.png", dpi=150)
    plt.close()


# ============================
# MAIN
# ============================

def main():
    ensure_plots_dir()
    df = load_csv(CSV_FILE)

    for dataset in DATASETS:
        print(f"→ Generating plots for {dataset}...")
        plot_times_per_dataset(df, dataset, PLOTS_DIR)
        plot_speedup(df, dataset, PLOTS_DIR)

    print("\n✔ All plots generated in 'plots/'")


if __name__ == "__main__":
    main()
