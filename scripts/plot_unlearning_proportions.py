"""Plot the six actual baseline evaluations in the style of analysis.ipynb.

Exports a three-panel figure and separate MSE/MAPE/Cost plots for each model,
as PNG and vector PDF, plus the full numerical results as CSV and JSON.
Missing or invalid values are errors; curves are never filled or interpolated.
Only load trusted local .npy results (the original format contains pickle).
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROPORTIONS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
MODELS = {
    'linear': ('linear', 'Linear'),
    'nn_conv': ('cnn', 'CNN'),
    'nn_mixer': ('mlp', 'MLP-Mixer'),
}
SPLITS = (('remain', 'Remain set', 'blue'),
          ('unlearn', 'Unlearn set', 'red'),
          ('test', 'Test set', 'green'))
METRICS = {
    'mse': (1e3, r'MSE ($\times 10^{-3}$)', '(a) MSE'),
    'mape': (1, 'MAPE (%)', '(b) MAPE'),
    'cost': (1e-3, r'Cost ($\times 10^{3}$)', '(c) Cost'),
}


def validate_metrics(metrics, source):
    normalized = {}
    for metric in METRICS:
        for method in ('ori', 'unlearn', 'retrain'):
            key = f'{metric}_{method}'
            normalized[key] = {}
            for split, _, _ in SPLITS:
                try:
                    value = np.asarray(metrics[key][split])
                    if value.shape != () or not np.isfinite(value) or value < 0:
                        raise ValueError('expected a finite nonnegative scalar')
                    normalized[key][split] = float(value)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f'{source}: invalid or missing {key}/{split}: {error}') from error
    return normalized


def load_results(directory):
    missing = [directory / model / f'unlearning_{proportion}.npy'
               for model in MODELS for proportion in PROPORTIONS
               if not (directory / model / f'unlearning_{proportion}.npy').is_file()]
    if missing:
        raise FileNotFoundError('Incomplete experiment; missing actual results:\n' + '\n'.join(map(str, missing)))
    records = []
    for model in MODELS:
        for proportion in PROPORTIONS:
            path = directory / model / f'unlearning_{proportion}.npy'
            metrics = validate_metrics(np.load(path, allow_pickle=True).item(), path)
            records.append(dict(model_type=model, unlearning_proportion=proportion,
                                source_file=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                metrics=metrics))
    manifest = directory / 'run-manifest.json'
    return dict(schema_version=1, description='Actual full-data case14 baseline evaluations; no inferred or paper-traced values.',
                plot_scaling=dict(mse=1e3, mape=1, cost=1e-3),
                run_manifest=json.loads(manifest.read_text()) if manifest.is_file() else None,
                evaluations={model: json.loads((directory / f'{model}-runs.json').read_text())
                             if (directory / f'{model}-runs.json').is_file() else None for model in MODELS},
                records=records)


def validate_records(payload):
    records = {}
    for row in payload['records']:
        key = (row['model_type'], row['unlearning_proportion'])
        if key in records:
            raise ValueError(f'Duplicate model/proportion result: {key}')
        records[key] = validate_metrics(row['metrics'], str(key))
    required = {(model, proportion) for model in MODELS for proportion in PROPORTIONS}
    if set(records) != required:
        raise ValueError(f'Expected exactly 18 model/proportion results; missing {required-set(records)}, extra {set(records)-required}')
    # The original model and fixed test set must be identical across proportions.
    for model in MODELS:
        for metric in METRICS:
            values = [records[model, p][f'{metric}_ori']['test'] for p in PROPORTIONS]
            if not np.allclose(values, values[0], rtol=1e-7, atol=1e-8):
                raise ValueError(f'{model}: original test {metric} changes across proportions: {values}')
    return records


def legend(ax):
    handles, labels = [], []
    for _, label, color in SPLITS:
        handles.extend([Line2D([], [], linestyle='none'),
                        Line2D([], [], color=color, linestyle=':', marker='o', markersize=3.5),
                        Line2D([], [], color=color, linestyle='-', marker='o', markersize=3.5)])
        labels.extend([label, 'Ori.Model', 'Unl.Model'])
    return ax.legend(handles, labels, ncol=3, loc='upper center', fontsize=7.5,
                     handlelength=1.6, handletextpad=0.4, columnspacing=1.2,
                     labelspacing=0.25, borderpad=0.4, framealpha=0.95)


def draw_panel(ax, records, model, metric, with_legend=False):
    factor, ylabel, caption = METRICS[metric]
    x = np.asarray(PROPORTIONS) * 100
    all_values = []
    for split, _, color in SPLITS:
        for method, style in (('ori', ':'), ('unlearn', '-')):
            values = [records[model, p][f'{metric}_{method}'][split] * factor for p in PROPORTIONS]
            all_values.extend(values)
            ax.plot(x, values, color=color, linestyle=style, marker='o',
                    markersize=3.5, linewidth=1.5)
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlim(3.8, 31.2)
    ax.set_xlabel('Unlearning proportion (%)', fontweight='bold', labelpad=3)
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=3)
    ax.grid(True, color='0.75', linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    # Preserve every actual point, rather than clipping to the paper's bounds.
    low, high = min(all_values), max(all_values)
    if metric == 'mse':
        ax.set_ylim(0, high * (1.36 if with_legend else 1.12))
    else:
        span = max(high-low, abs(high)*0.05, 1e-6)
        ax.set_ylim(max(0, low-span*0.15), high+span*(0.45 if with_legend else 0.15))
    if with_legend:
        legend(ax)
    ax.text(0.5, -0.38, caption, transform=ax.transAxes,
            ha='center', va='top', fontsize=11)


def save_figure(fig, stem):
    fig.savefig(stem.with_suffix('.png'), dpi=300, facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white', metadata={'Creator': 'Matplotlib, actual TX-123BT experiment results'})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--results-dir', type=Path, default=ROOT / 'outputs/paper-proportions')
    source.add_argument('--metrics-json', type=Path, help='Redraw from the exported complete numerical results without .npy files.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'figs')
    args = parser.parse_args()
    payload = json.loads(args.metrics_json.read_text()) if args.metrics_json else load_results(args.results_dir.resolve())
    records = validate_records(payload)
    payload['plotter'] = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             matplotlib_version=matplotlib.__version__, numpy_version=np.__version__)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'pdf.fonttype': 42})
    for model, (folder, name) in MODELS.items():
        destination = output / folder
        destination.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
        fig.subplots_adjust(left=0.058, right=0.991, bottom=0.29, top=0.86, wspace=0.32)
        fig.suptitle(name, fontsize=13, fontweight='bold', y=0.98)
        for index, (ax, metric) in enumerate(zip(axes, METRICS)):
            draw_panel(ax, records, model, metric, with_legend=index == 0)
        save_figure(fig, destination / 'unlearning_metrics')
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(4.5, 3.9))
            fig.subplots_adjust(left=0.17, right=0.97, bottom=0.28, top=0.86)
            fig.suptitle(name, fontsize=12, fontweight='bold', y=0.98)
            draw_panel(ax, records, model, metric, with_legend=True)
            save_figure(fig, destination / metric)
        print(f'{name}: three-panel and separate plots saved in {destination}')

    (output / 'unlearning_metrics.json').write_text(json.dumps(payload, indent=2) + '\n')
    columns = ['model_type', 'unlearning_proportion', 'split'] + [f'{metric}_{method}'
               for metric in METRICS for method in ('ori', 'unlearn', 'retrain')]
    with (output / 'unlearning_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for (model, proportion), metrics in records.items():
            for split, _, _ in SPLITS:
                writer.writerow(dict(model_type=model, unlearning_proportion=proportion, split=split,
                                     **{key: metrics[key][split] for key in columns[3:]}))
    print(f'Validated 18 experiments, 486 scalar metrics; exported 54 CSV rows in {output}')


if __name__ == '__main__':
    main()
