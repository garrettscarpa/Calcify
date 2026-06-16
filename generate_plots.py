"""
plot_pre_post_frequency.py
--------------------------
Reads calcium imaging output CSVs and plots pre vs post peak frequency
as a bar graph, with individual ROI data points overlaid.

Usage:
    python plot_pre_post_frequency.py --stats path/to/F_Individual_ROI_Statistics_timelocked.csv

    # Or point to a folder and it will find the file automatically:
    python plot_pre_post_frequency.py --folder path/to/suite2p/plane0/

    # To compare multiple recordings:
    python plot_pre_post_frequency.py --folder /path/to/recording1 /path/to/recording2
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats



# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
folder_path = '/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/Nodose/Data/4_17_26/TSeries-04172026-mouse2-DCZ-000'





# ------------------------------------------------------------------------------
# File discovery
# ------------------------------------------------------------------------------

def find_timelocked_stats(folder):
    """Search folder recursively for the timelocked ROI statistics CSV."""
    pattern = os.path.join(folder, '**', '*_Individual_ROI_Statistics_timelocked.csv')
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"No timelocked statistics CSV found under: {folder}\n"
            "Expected a file matching *_Individual_ROI_Statistics_timelocked.csv"
        )
    if len(matches) > 1:
        print(f"[WARN] Multiple timelocked files found, using first:\n  {matches[0]}")
    return matches[0]


# ------------------------------------------------------------------------------
# Data loading and validation
# ------------------------------------------------------------------------------

def load_timelocked_stats(csv_path):
    """
    Load and validate timelocked ROI statistics.
    Expected columns: ROI, time_period, peak_count, duration_min, freq_PeaksPerMin
    """
    df = pd.read_csv(csv_path)

    required = {'ROI', 'time_period', 'freq_PeaksPerMin'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}\nFound: {list(df.columns)}")

    periods = df['time_period'].unique()
    print(f"[INFO] Time periods found: {sorted(periods)}")

    if 'pre' not in periods or 'post' not in periods:
        raise ValueError(
            f"Expected 'pre' and 'post' in time_period column. Found: {sorted(periods)}"
        )

    return df


# ------------------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------------------

def compute_summary(df):
    """
    Compute mean, SEM, and paired t-test for pre vs post frequency.
    Returns summary dict and aligned pre/post arrays per ROI.
    """
    pre_df  = df[df['time_period'] == 'pre'].set_index('ROI')['freq_PeaksPerMin']
    post_df = df[df['time_period'] == 'post'].set_index('ROI')['freq_PeaksPerMin']

    # Align on common ROIs
    common_rois = pre_df.index.intersection(post_df.index)
    if len(common_rois) == 0:
        raise ValueError("No ROIs appear in both pre and post periods.")

    pre_vals  = pre_df.loc[common_rois].values.astype(float)
    post_vals = post_df.loc[common_rois].values.astype(float)

    n = len(common_rois)
    t_stat, p_val = stats.ttest_rel(pre_vals, post_vals)

    summary = {
        'rois':      common_rois.tolist(),
        'pre_vals':  pre_vals,
        'post_vals': post_vals,
        'pre_mean':  np.mean(pre_vals),
        'post_mean': np.mean(post_vals),
        'pre_sem':   np.std(pre_vals, ddof=1) / np.sqrt(n),
        'post_sem':  np.std(post_vals, ddof=1) / np.sqrt(n),
        'n':         n,
        't_stat':    t_stat,
        'p_val':     p_val,
    }
    summary['n_responsive'] = len(df['ROI'].unique())

    return summary


def p_to_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------

def plot_frequency(summary, title='', save_path=None):
    """
    Bar graph of mean pre vs post frequency with SEM error bars,
    individual ROI data points, and connecting lines.
    """
    fig, ax = plt.subplots(figsize=(4.5, 5.5))

    x      = [0, 1]
    means  = [summary['pre_mean'],  summary['post_mean']]
    sems   = [summary['pre_sem'],   summary['post_sem']]
    colors = ['#4C8BB5', '#E06B4A']
    labels = ['Pre', 'Post']

    # --- Bars ---
    bars = ax.bar(x, means, yerr=sems, width=0.45,
                  color=colors, alpha=0.75, capsize=6,
                  error_kw=dict(elinewidth=1.5, ecolor='black', capthick=1.5),
                  zorder=2)

    # --- Individual ROI lines and points ---
    jitter = np.random.uniform(-0.07, 0.07, size=summary['n'])
    for i, (pre, post) in enumerate(zip(summary['pre_vals'], summary['post_vals'])):
        xp = [0 + jitter[i], 1 + jitter[i]]
        ax.plot(xp, [pre, post], color='gray', alpha=0.4, linewidth=0.8, zorder=3)
        ax.plot(0 + jitter[i], pre,  'o', color=colors[0], markersize=5,
                alpha=0.8, markeredgecolor='white', markeredgewidth=0.5, zorder=4)
        ax.plot(1 + jitter[i], post, 'o', color=colors[1], markersize=5,
                alpha=0.8, markeredgecolor='white', markeredgewidth=0.5, zorder=4)

    # --- Significance bracket ---
    y_max = max(summary['pre_vals'].max(), summary['post_vals'].max())
    y_top = y_max * 1.15
    stars = p_to_stars(summary['p_val'])
    bracket_h = y_max * 0.04

    ax.plot([0, 0, 1, 1],
            [y_top, y_top + bracket_h, y_top + bracket_h, y_top],
            color='black', linewidth=1.2)
    ax.text(0.5, y_top + bracket_h * 1.3, stars,
            ha='center', va='bottom', fontsize=13,
            fontweight='bold' if stars != 'ns' else 'normal')

    # --- Labels and formatting ---
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Peak Frequency (peaks / min)', fontsize=11)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, (y_top + bracket_h) * 1.2)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=10)

    p_str = f'p = {summary["p_val"]:.4f}' if summary['p_val'] >= 0.0001 else 'p < 0.0001'

    n_total_str = f' / {summary["n_total"]}' if summary.get('n_total') is not None else ''
    subtitle = (f'n = {summary["n_total"]}{n_total_str and ""} total ROIs  |  '
                f'{summary["n_with_peaks"]} ROI with peaks  |  '
                f'paired t-test  |  {p_str}')

    ax.set_title(title + ('\n' if title else '') + subtitle, fontsize=9, color='#555555')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Figure saved to: {save_path}")

    plt.show()
    return fig


# ------------------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------------------

def print_summary(summary, label=''):
    print(f"\n{'='*50}")
    if label:
        print(f"  {label}")
    print(f"{'='*50}")
    print(f"  ROIs analysed : {summary['n']}")
    print(f"  Pre  freq     : {summary['pre_mean']:.4f} ± {summary['pre_sem']:.4f} peaks/min (mean ± SEM)")
    print(f"  Post freq     : {summary['post_mean']:.4f} ± {summary['post_sem']:.4f} peaks/min (mean ± SEM)")
    print(f"  Paired t-test : t = {summary['t_stat']:.3f},  p = {summary['p_val']:.4f}  ({p_to_stars(summary['p_val'])})")
    print(f"{'='*50}\n")

    # Per-ROI table
    print(f"  {'ROI':<12} {'Pre (peaks/min)':>18} {'Post (peaks/min)':>18} {'Δ':>10}")
    print(f"  {'-'*60}")
    for roi, pre, post in zip(summary['rois'], summary['pre_vals'], summary['post_vals']):
        delta = post - pre
        sign  = '+' if delta >= 0 else ''
        print(f"  {str(roi):<12} {pre:>18.4f} {post:>18.4f} {sign+f'{delta:.4f}':>10}")
    print()


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def process_one(csv_path, save_dir=None):
    parts = csv_path.replace('\\', '/').split('/')
    folder_parts = [p for p in parts if p]
    try:
        label = f"{folder_parts[-5]} : {folder_parts[-4]}"
    except IndexError:
        label = os.path.basename(os.path.dirname(csv_path))
    print(f"\n[INFO] Loading: {csv_path}")

    df = load_timelocked_stats(csv_path)

    # --- Count total ROIs from the main filtered peaks CSV ---
    export_dir = os.path.dirname(csv_path)
    filtered_peaks_files = glob.glob(os.path.join(export_dir, '*_filtered_peaks.csv'))
    # Exclude the timelocked version
    filtered_peaks_files = [f for f in filtered_peaks_files if 'timelocked' not in f]
    
    n_total = None
    n_with_peaks = None
    if filtered_peaks_files:
        peaks_df = pd.read_csv(filtered_peaks_files[0])
        if 'cell_id' in peaks_df.columns:
            n_with_peaks = peaks_df['cell_id'].nunique()
    
    # Also try Individual_ROI_Statistics for total ROI count
    stats_files = glob.glob(os.path.join(export_dir, '*_Individual_ROI_Statistics.csv'))
    stats_files = [f for f in stats_files if 'timelocked' not in f]
    if stats_files:
        stats_df = pd.read_csv(stats_files[0])
        if 'ROI' in stats_df.columns:
            n_total = len(stats_df)

    summary = compute_summary(df)
    
    # Override n counts with more accurate values
    if n_total is not None:
        summary['n_total'] = n_total
    else:
        summary['n_total'] = None
    if n_with_peaks is not None:
        summary['n_with_peaks'] = n_with_peaks
    else:
        summary['n_with_peaks'] = summary['n_responsive']

    print_summary(summary, label=label)

    save_path = None
    if save_dir:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        save_path = os.path.join(save_dir, f"{base}_pre_post_frequency.png")

    plot_frequency(summary, title=label, save_path=save_path)
    return summary

def main():
    folders = [folder_path]
    csv_paths = [find_timelocked_stats(f) for f in folders]

    save_figures = False  # Set True to save a PNG next to each CSV
# ------------------------------------------------------------------------------

    for csv_path in csv_paths:
        save_dir = os.path.dirname(csv_path) if save_figures else None
        process_one(csv_path, save_dir=save_dir)


if __name__ == '__main__':
    main()