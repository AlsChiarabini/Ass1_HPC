import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# configurazione per le due cartelle dato che compiliamo con GCC e con CLANG.
CSV_FILES = {
    "atax_timings.csv": {"label": "GCC", "dir": "gcc"},
    "atax_timings_clang.csv": {"label": "Clang", "dir": "clang"}
}

PLOTS_DIR = Path("plots")
DATASETS = ["MINI_DATASET", "SMALL_DATASET", "STANDARD_DATASET", "LARGE_DATASET", "EXTRALARGE_DATASET"]
BASELINE_KERNEL = "SEQUENTIAL"

# Selezione dello stile di seaborn
sns.set_theme(style="whitegrid", palette="husl")
COLORS = sns.color_palette("husl", 8)


def ensure_plots_dir():
    #Creiamo la cartella plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv_data(filepath):
    #Carichiamo il csv
    df = pd.read_csv(filepath)
    
    # Criterio di arresto nel for = quando troviamo la keyword "Average_Time"
    if 'Average_Time(s)' in df.columns:
        # This means the averages are in a separate section, filter them out
        df = df[df['Dataset'].notna()]
    
    return df


def plot_execution_times_per_dataset(df, dataset, compiler_label, output_dir):
    """Create bar chart comparing kernel execution times for a specific dataset."""
    dataset_data = df[df['Dataset'] == dataset].copy()
    
    if dataset_data.empty:
        print(f"Warning: No data found for dataset {dataset}")
        return
    
    # Ordinamento sulla colonna Times(s)
    dataset_data = dataset_data.sort_values('Time(s)')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=dataset_data, x='Kernel', y='Time(s)', ax=ax, hue='Kernel', palette="husl", legend=False)
    
    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel('Execution Time (s)', fontsize=12)
    ax.set_xlabel('Kernel Version', fontsize=12)
    ax.set_title(f'Execution Time Comparison - {dataset.replace("_", " ").title()} ({compiler_label})', 
                 fontsize=14, fontweight='bold')
    
    # Aggiunta dei valori
    for container in ax.containers:
        ax.bar_label(container, fmt='%.6f', fontsize=8, padding=3)
    
    plt.tight_layout()
    
    output_file = output_dir / f"{dataset.lower()}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_speedup_curves_per_dataset(df, dataset, compiler_label, output_dir):
    #Creiamo la curva di speeup
    dataset_data = df[df['Dataset'] == dataset].copy()
        
    baseline_row = dataset_data[dataset_data['Kernel'] == BASELINE_KERNEL]
    baseline_time = baseline_row['Time(s)'].values[0]
    
    # Calcolo dello speeup
    dataset_data['Speedup'] = baseline_time / dataset_data['Time(s)']
    #ordiniamo in base allo speeup
    dataset_data = dataset_data.sort_values('Speedup')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=dataset_data, x='Kernel', y='Speedup', marker='o', 
                 linewidth=2.5, markersize=10, ax=ax, color='#2ecc71')
    
    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel('Speedup (vs Sequential)', fontsize=12)
    ax.set_xlabel('Kernel Version', fontsize=12)
    ax.set_title(f'Speedup Curve - {dataset.replace("_", " ").title()} ({compiler_label})', 
                 fontsize=14, fontweight='bold')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (1×)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Aggiungiamo le lable
    for x, y in enumerate(dataset_data['Speedup'].values):
        ax.text(x, y + 0.05, f'{y:.2f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / f"{dataset.lower()}_speedup.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()





def main():
    ensure_plots_dir()
    
    for csv_file, config in CSV_FILES.items():
        csv_path = Path(csv_file)
        
        # Creiamo le directory per gcc e per clang
        compiler_dir = PLOTS_DIR / config["dir"]
        compiler_dir.mkdir(parents=True, exist_ok=True)        
    
        df = load_csv_data(csv_path)
        
        # Iteriamo sui vari dataset
        for dataset in DATASETS:
            print(f"\nGenerating plots for {dataset}...")
            plot_execution_times_per_dataset(df, dataset, config['label'], compiler_dir)
            plot_speedup_curves_per_dataset(df, dataset, config['label'], compiler_dir)
        
        print("\n All plots generated successfully")

if __name__ == "__main__":
    main()
