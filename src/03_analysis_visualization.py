import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from matplotlib.patches import Patch
from pathlib import Path
import zipfile
import warnings
import glob
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

class Config:
    RESULTS_DIR = './results/'
    OUTPUT_DIR = './outputs/'
    CSV_DIR = './outputs/csv_tables/'
    FIGURES_DIR = './outputs/figures/'
    INDIVIDUAL_DIR = './outputs/figures/individual_models/'
    COMBINED_DIR = './outputs/figures/combined/'
    ZIP_NAME = 'calibration_study_complete.zip'
    ECE_BINS = 10
    ENTROPY_BINS = 20
    FIGURE_FORMAT = 'png'
    HIGH_DPI = 300

    def __init__(self):
        for dir_path in [self.CSV_DIR, self.FIGURES_DIR, self.INDIVIDUAL_DIR, self.COMBINED_DIR]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

config = Config()

class Metrics:

    @staticmethod
    def compute_ece(conf, correct, bins=10):
        bin_boundaries = np.linspace(0, 1, bins + 1)
        ece = 0.0
        for i in range(bins):
            in_bin = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
            if in_bin.sum() == 0:
                continue
            bin_acc = correct[in_bin].mean()
            bin_conf = conf[in_bin].mean()
            bin_weight = in_bin.sum() / len(conf)
            ece += bin_weight * abs(bin_acc - bin_conf)
        return ece

    @staticmethod
    def compute_mce(conf, correct, bins=10):
        bin_boundaries = np.linspace(0, 1, bins + 1)
        mce = 0.0
        for i in range(bins):
            in_bin = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
            if in_bin.sum() == 0:
                continue
            bin_acc = correct[in_bin].mean()
            bin_conf = conf[in_bin].mean()
            mce = max(mce, abs(bin_acc - bin_conf))
        return mce

    @staticmethod
    def compute_confidence_entropy(conf, bins=20):
        (hist, _) = np.histogram(conf, bins=bins, range=(0, 1), density=True)
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        hist = hist[hist > 0]
        return entropy(hist) if len(hist) > 0 else 0.0

    @staticmethod
    def get_ece_breakdown(conf, correct, bins=10):
        bin_boundaries = np.linspace(0, 1, bins + 1)
        bin_data = []
        for i in range(bins):
            in_bin = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_center = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2
                bin_acc = correct[in_bin].mean()
                bin_conf = conf[in_bin].mean()
                gap = bin_conf - bin_acc
                count = in_bin.sum()
                bin_data.append({'bin_center': bin_center, 'bin_start': bin_boundaries[i], 'bin_end': bin_boundaries[i + 1], 'accuracy': bin_acc, 'confidence': bin_conf, 'gap': gap, 'count': count})
        return pd.DataFrame(bin_data)

    @staticmethod
    def get_confidence_distribution(conf, bins=20):
        (hist, bin_edges) = np.histogram(conf, bins=bins, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return pd.DataFrame({'bin_center': bin_centers, 'count': hist, 'density': hist / hist.sum()})

    @staticmethod
    def compute_all_metrics(df):
        conf = df['confidence'].values
        correct = df['correct'].values
        return {'accuracy': correct.mean(), 'avg_confidence': conf.mean(), 'overconfidence_gap': conf.mean() - correct.mean(), 'ece': Metrics.compute_ece(conf, correct, config.ECE_BINS), 'mce': Metrics.compute_mce(conf, correct, config.ECE_BINS), 'brier_score': np.mean((conf - correct) ** 2), 'confidence_entropy': Metrics.compute_confidence_entropy(conf, config.ENTROPY_BINS), 'n_samples': len(df), 'n_successful': df['parse_success'].sum() if 'parse_success' in df.columns else len(df)}

def load_data():
    print('\nLoading data...')
    
    combined_path = Path(f'{config.RESULTS_DIR}evaluation_results_combined.csv')
    combined_gz_path = Path(f'{config.RESULTS_DIR}evaluation_results_combined.csv.gz')
    
    if combined_gz_path.exists():
        print(f'  Loading: {combined_gz_path.name}')
        df = pd.read_csv(combined_gz_path, compression='gzip')
    elif combined_path.exists():
        print(f'  Loading: {combined_path.name}')
        df = pd.read_csv(combined_path)
    else:
        # Fallback to individual model files (support both .csv and .csv.gz)
        csv_files = glob.glob(f'{config.RESULTS_DIR}*_results.csv*')
        if not csv_files:
            raise FileNotFoundError(f'No results files found in {config.RESULTS_DIR}')
        
        df_list = []
        for csv_file in csv_files:
            # Skip summary just in case
            if "model_summary" in csv_file or "combined" in csv_file:
                continue
            print(f'  Loading: {Path(csv_file).name}')
            temp_df = pd.read_csv(csv_file)
            if 'model' not in temp_df.columns:
                model_name = Path(csv_file).name.split('_results')[0].replace('_', '/')
                temp_df['model'] = model_name
            df_list.append(temp_df)
        df = pd.concat(df_list, ignore_index=True)
        
    print(f'✓ Loaded {len(df)} rows')
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Datasets: {df['dataset'].nunique()}")
    print(f"  Questions: {df['id'].nunique()}")
    df = df[(df['confidence'] >= 0) & (df['confidence'] <= 1) & df['prediction'].notna()]
    print(f'✓ Cleaned to {len(df)} rows')
    return df

def export_csvs(df):
    print('\n' + '=' * 80)
    print('EXPORTING CSV TABLES')
    print('=' * 80)
    csv_files = []
    print('\n1. Computing calibration metrics per model...')
    metrics_list = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        metrics = Metrics.compute_all_metrics(model_data)
        metrics['model'] = model
        metrics_list.append(metrics)
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df[['model', 'accuracy', 'avg_confidence', 'overconfidence_gap', 'ece', 'mce', 'brier_score', 'confidence_entropy', 'n_samples', 'n_successful']]
    path = f'{config.CSV_DIR}calibration_metrics.csv'
    metrics_df.to_csv(path, index=False)
    csv_files.append(path)
    print(f'   ✓ Saved: calibration_metrics.csv')
    print('\n2. Computing ECE breakdown per model...')
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        model_short = model.replace('/', '_').replace('-', '_')
        ece_breakdown = Metrics.get_ece_breakdown(model_data['confidence'].values, model_data['correct'].values, config.ECE_BINS)
        path = f'{config.CSV_DIR}ece_breakdown_{model_short}.csv'
        ece_breakdown.to_csv(path, index=False)
        csv_files.append(path)
    print(f"   ✓ Saved: {len(df['model'].unique())} ECE breakdown files")
    print('\n3. Computing confidence distribution per model...')
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        model_short = model.replace('/', '_').replace('-', '_')
        conf_dist = Metrics.get_confidence_distribution(model_data['confidence'].values, config.ENTROPY_BINS)
        path = f'{config.CSV_DIR}confidence_distribution_{model_short}.csv'
        conf_dist.to_csv(path, index=False)
        csv_files.append(path)
    print(f"   ✓ Saved: {len(df['model'].unique())} confidence distribution files")
    print('\n4. Computing per-dataset performance...')
    dataset_perf = []
    for model in df['model'].unique():
        for dataset in df['dataset'].unique():
            subset = df[(df['model'] == model) & (df['dataset'] == dataset)]
            if len(subset) == 0:
                continue
            metrics = Metrics.compute_all_metrics(subset)
            metrics['model'] = model
            metrics['dataset'] = dataset
            dataset_perf.append(metrics)
    dataset_df = pd.DataFrame(dataset_perf)
    dataset_df = dataset_df[['model', 'dataset', 'accuracy', 'avg_confidence', 'overconfidence_gap', 'ece', 'n_samples']]
    path = f'{config.CSV_DIR}per_dataset_performance.csv'
    dataset_df.to_csv(path, index=False)
    csv_files.append(path)
    print(f'   ✓ Saved: per_dataset_performance.csv')
    print(f'\n:✓ Exported {len(csv_files)} CSV files')
    return csv_files

def plot_individual_reliability_diagrams(df):
    print('\n' + '=' * 80)
    print('CREATING INDIVIDUAL RELIABILITY DIAGRAMS')
    print('=' * 80)
    figure_files = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        model_short = model.replace('/', '_').replace('-', '_')
        print(f"\nPlotting: {model.split('/')[-1]}")
        (fig, ax) = plt.subplots(figsize=(10, 10))
        conf = model_data['confidence'].values
        correct = model_data['correct'].values
        bin_boundaries = np.linspace(0, 1, config.ECE_BINS + 1)
        bin_centers = []
        bin_accs = []
        bin_counts = []
        for i in range(config.ECE_BINS):
            in_bin = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
                bin_accs.append(correct[in_bin].mean())
                bin_counts.append(in_bin.sum())
        ax.plot(bin_centers, bin_accs, 'o-', linewidth=3, markersize=12, label='Model', color='#2E86AB', zorder=3)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.5, zorder=2)
        for (x, y) in zip(bin_centers, bin_accs):
            ax.plot([x, x], [x, y], 'r-', alpha=0.3, linewidth=2, zorder=1)
        ax2 = ax.twinx()
        ax2.bar(bin_centers, bin_counts, width=0.08, alpha=0.2, color='gray', label='Sample Count')
        ax2.set_ylabel('Sample Count', fontsize=12)
        ax2.legend(loc='upper left')
        ax.set_xlabel('Confidence', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        metrics = Metrics.compute_all_metrics(model_data)
        title = f"{model.split('/')[-1]}\n"
        title += f"ECE={metrics['ece']:.3f}, Acc={metrics['accuracy']:.3f}, "
        title += f"Gap={metrics['overconfidence_gap']:+.3f}"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()
        path = f'{config.INDIVIDUAL_DIR}reliability_{model_short}.{config.FIGURE_FORMAT}'
        plt.savefig(path, dpi=config.HIGH_DPI, bbox_inches='tight')
        plt.close(fig)
        figure_files.append(path)
    print(f'\n✓ Created {len(figure_files)} reliability diagrams')
    return figure_files

def plot_combined_reliability_diagram(df):
    print('\n' + '=' * 80)
    print('CREATING COMBINED RELIABILITY DIAGRAM')
    print('=' * 80)
    (fig, ax) = plt.subplots(figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, df['model'].nunique()))
    for (idx, model) in enumerate(df['model'].unique()):
        model_data = df[df['model'] == model]
        conf = model_data['confidence'].values
        correct = model_data['correct'].values
        bin_boundaries = np.linspace(0, 1, config.ECE_BINS + 1)
        bin_centers = []
        bin_accs = []
        for i in range(config.ECE_BINS):
            in_bin = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
                bin_accs.append(correct[in_bin].mean())
        model_short = model.split('/')[-1][:25]
        metrics = Metrics.compute_all_metrics(model_data)
        label = f"{model_short} (ECE={metrics['ece']:.3f})"
        ax.plot(bin_centers, bin_accs, 'o-', linewidth=2, markersize=8, label=label, color=colors[idx], alpha=0.8)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=3, label='Perfect Calibration', alpha=0.7)
    ax.set_xlabel('Confidence', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Reliability Diagram - All Models', fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    path = f'{config.COMBINED_DIR}reliability_all_models.{config.FIGURE_FORMAT}'
    plt.savefig(path, dpi=config.HIGH_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'✓ Saved: reliability_all_models.{config.FIGURE_FORMAT}')
    return path

def plot_metric_comparison(df, metric_name, metric_col, ylabel, title):
    metrics_list = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        metrics = Metrics.compute_all_metrics(model_data)
        metrics['model'] = model.split('/')[-1][:30]
        metrics_list.append(metrics)
    metrics_df = pd.DataFrame(metrics_list).sort_values(metric_col, ascending=False)
    (fig, ax) = plt.subplots(figsize=(12, max(6, len(metrics_df) * 0.4)))
    colors = ['green' if v <= 0.15 else 'orange' if v <= 0.3 else 'red' for v in metrics_df[metric_col]]
    ax.barh(metrics_df['model'], metrics_df[metric_col], color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = f'{config.COMBINED_DIR}{metric_name}_comparison.{config.FIGURE_FORMAT}'
    plt.savefig(path, dpi=config.HIGH_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'✓ Saved: {metric_name}_comparison.{config.FIGURE_FORMAT}')
    return path

def plot_all_metric_comparisons(df):
    print('\n' + '=' * 80)
    print('CREATING METRIC COMPARISON CHARTS')
    print('=' * 80)
    figure_files = []
    metrics_to_plot = [('ece', 'ece', 'ECE', 'Expected Calibration Error (ECE) - All Models'), ('mce', 'mce', 'MCE', 'Maximum Calibration Error (MCE) - All Models'), ('avg_confidence', 'avg_confidence', 'Average Confidence', 'Average Confidence - All Models'), ('brier_score', 'brier_score', 'Brier Score', 'Brier Score - All Models'), ('overconfidence_gap', 'overconfidence_gap', 'Overconfidence Gap', 'Overconfidence Gap - All Models'), ('confidence_entropy', 'confidence_entropy', 'Confidence Entropy', 'Confidence Entropy - All Models')]
    for (metric_name, metric_col, ylabel, title) in metrics_to_plot:
        print(f'\nPlotting: {title}')
        path = plot_metric_comparison(df, metric_name, metric_col, ylabel, title)
        figure_files.append(path)
    print(f'\n✓ Created {len(figure_files)} comparison charts')
    return figure_files

def plot_ece_breakdown(df):
    print('\n' + '=' * 80)
    print('CREATING ECE BREAKDOWN CHARTS')
    print('=' * 80)
    figure_files = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        model_short = model.replace('/', '_').replace('-', '_')
        print(f"\nPlotting: {model.split('/')[-1]}")
        conf = model_data['confidence'].values
        correct = model_data['correct'].values
        ece_df = Metrics.get_ece_breakdown(conf, correct, config.ECE_BINS)
        (fig, ax) = plt.subplots(figsize=(12, 6))
        colors = ['red' if gap > 0 else 'blue' for gap in ece_df['gap']]
        bars = ax.bar(ece_df['bin_center'], ece_df['gap'], width=0.08, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(0, color='k', linestyle='-', linewidth=1)
        for (bar, count) in zip(bars, ece_df['count']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'n={count}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        ax.set_xlabel('Confidence Bin Center', fontsize=12, fontweight='bold')
        ax.set_ylabel('Calibration Gap (Confidence - Accuracy)', fontsize=12, fontweight='bold')
        ax.set_title(f"ECE Breakdown - {model.split('/')[-1]}", fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Overconfident'), Patch(facecolor='blue', alpha=0.7, label='Underconfident')]
        ax.legend(handles=legend_elements, loc='best')
        plt.tight_layout()
        path = f'{config.INDIVIDUAL_DIR}ece_breakdown_{model_short}.{config.FIGURE_FORMAT}'
        plt.savefig(path, dpi=config.HIGH_DPI, bbox_inches='tight')
        plt.close(fig)
        figure_files.append(path)
    print(f'\n✓ Created {len(figure_files)} ECE breakdown charts')
    return figure_files

def plot_confidence_distributions(df):
    print('\n' + '=' * 80)
    print('CREATING CONFIDENCE DISTRIBUTION CHARTS')
    print('=' * 80)
    figure_files = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        model_short = model.replace('/', '_').replace('-', '_')
        print(f"\nPlotting: {model.split('/')[-1]}")
        (fig, axes) = plt.subplots(1, 2, figsize=(16, 6))
        conf_dist = Metrics.get_confidence_distribution(model_data['confidence'].values, config.ENTROPY_BINS)
        axes[0].bar(conf_dist['bin_center'], conf_dist['count'], width=1 / config.ENTROPY_BINS, alpha=0.7, color='purple', edgecolor='black')
        axes[0].set_xlabel('Confidence', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
        metrics = Metrics.compute_all_metrics(model_data)
        axes[0].set_title(f"Confidence Distribution (Entropy={metrics['confidence_entropy']:.3f})", fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        correct_conf = model_data[model_data['correct'] == 1]['confidence']
        incorrect_conf = model_data[model_data['correct'] == 0]['confidence']
        axes[1].hist(correct_conf, bins=20, alpha=0.6, label='Correct', color='green', density=True)
        axes[1].hist(incorrect_conf, bins=20, alpha=0.6, label='Incorrect', color='red', density=True)
        axes[1].set_xlabel('Confidence', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Density', fontsize=12, fontweight='bold')
        axes[1].set_title('Confidence by Correctness', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        fig.suptitle(f"{model.split('/')[-1]} - Confidence Analysis", fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = f'{config.INDIVIDUAL_DIR}confidence_dist_{model_short}.{config.FIGURE_FORMAT}'
        plt.savefig(path, dpi=config.HIGH_DPI, bbox_inches='tight')
        plt.close(fig)
        figure_files.append(path)
    print(f'\n✓ Created {len(figure_files)} confidence distribution charts')
    return figure_files

def create_zip_package(csv_files, figure_files):
    print('\n' + '=' * 80)
    print('CREATING ZIP PACKAGE')
    print('=' * 80)
    zip_path = config.OUTPUT_DIR + config.ZIP_NAME
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        print('\nAdding CSV files...')
        for csv_file in csv_files:
            arcname = f'csv_tables/{Path(csv_file).name}'
            zipf.write(csv_file, arcname)
        print(f'✓ Added {len(csv_files)} CSV files')
        print('\nAdding figure files...')
        for fig_file in figure_files:
            if 'individual' in fig_file:
                arcname = f'figures/individual_models/{Path(fig_file).name}'
            else:
                arcname = f'figures/combined/{Path(fig_file).name}'
            zipf.write(fig_file, arcname)
        print(f'✓ Added {len(figure_files)} figure files')
    size_mb = Path(zip_path).stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 80}")
    print(f'✓ Created ZIP package: {config.ZIP_NAME}')
    print(f'  Size: {size_mb:.2f} MB')
    print(f'  CSV files: {len(csv_files)}')
    print(f'  Figure files: {len(figure_files)}')
    print(f'  Total files: {len(csv_files) + len(figure_files)}')
    print(f"{'=' * 80}")
    return zip_path

def dataset_heatmap(df):
    rows = []
    for m in df.model.unique():
        for d in df.dataset.unique():
            sub = df[(df.model == m) & (df.dataset == d)]
            if len(sub) == 0:
                continue
            rows.append({
                "model": m.split("/")[-1],
                "dataset": d,
                "accuracy": sub.correct.mean()
            })
    dfh = pd.DataFrame(rows)
    pivot = dfh.pivot(index="model", columns="dataset", values="accuracy")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=.5,
        ax=ax
    )
    ax.set_title("Model Accuracy Across Datasets")
    path = f"{config.COMBINED_DIR}dataset_heatmap.png"
    fig.savefig(path, dpi=config.HIGH_DPI, bbox_inches='tight')
    plt.close()
    return path

def main():
    print('=' * 80)
    print('LLM CONFIDENCE CALIBRATION - COMPREHENSIVE VISUALIZATION')
    print('=' * 80)
    df = load_data()
    csv_files = export_csvs(df)
    all_figures = []
    all_figures.extend(plot_individual_reliability_diagrams(df))
    all_figures.append(plot_combined_reliability_diagram(df))
    all_figures.extend(plot_all_metric_comparisons(df))
    all_figures.append(dataset_heatmap(df))
    all_figures.extend(plot_ece_breakdown(df))
    all_figures.extend(plot_confidence_distributions(df))
    zip_path = create_zip_package(csv_files, all_figures)
    print('\n' + '=' * 80)
    print('VISUALIZATION COMPLETE!')
    print('=' * 80)

if __name__ == "__main__":
    main()