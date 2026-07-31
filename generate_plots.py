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
folder_path = '/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/Calcium Imaging/Nodose/Data/4_9_26/TSeries-04092026-1418-008/suite2p/plane0'

# Pre/post windows in SECONDS, matching what was entered in Calcify.
# Peaks are assigned by peak_time; used to aggregate AUC from *_filtered_peaks.csv.
PRE_RANGE  = (0, 300)
POST_RANGE = (300, 773)




METRIC = 'freq_PeaksPerMin'   # 'freq_PeaksPerMin' | 'mean_auc' | 'total_auc' | 'auc_per_min' | 'mean_amplitude' | 'max_amplitude'
METRIC = 'mean_amplitude'

METRIC_LABELS = {
    'freq_PeaksPerMin': 'Peak Frequency (peaks / min)',
    'mean_auc':         'Mean Peak AUC (ΔF/F·s)',
    'total_auc':        'Total AUC (ΔF/F·s)',
    'auc_per_min':      'AUC per Minute (ΔF/F·s / min)',
    'mean_amplitude':   'Mean Peak Amplitude (ΔF/F)',
    'max_amplitude':    'Max Peak Amplitude (ΔF/F)',
}

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

    required = {'ROI', 'time_period', METRIC}
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
def _one_metric(df, metric):
    pre  = df[df['time_period'] == 'pre'].set_index('ROI')[metric].dropna().values.astype(float)
    post = df[df['time_period'] == 'post'].set_index('ROI')[metric].dropna().values.astype(float)
    if len(pre) == 0 or len(post) == 0:
        return None
    n_pre, n_post = len(pre), len(post)
    if n_pre > 1 and n_post > 1:
        t, p = stats.ttest_ind(pre, post, equal_var=False)
    else:
        t, p = np.nan, np.nan
    return {
        'pre_vals': pre, 'post_vals': post,
        'pre_mean': np.mean(pre), 'post_mean': np.mean(post),
        'pre_sem': np.std(pre, ddof=1) / np.sqrt(n_pre) if n_pre > 1 else np.nan,
        'post_sem': np.std(post, ddof=1) / np.sqrt(n_post) if n_post > 1 else np.nan,
        'n_pre': n_pre, 'n_post': n_post, 't_stat': t, 'p_val': p,
    }


def compute_summary(df):
    """
    Compute mean, SEM, and paired t-test for pre vs post frequency.
    Returns summary dict and aligned pre/post arrays per ROI.
    """
    pre_df  = df[df['time_period'] == 'pre'].set_index('ROI')[METRIC].dropna()
    post_df = df[df['time_period'] == 'post'].set_index('ROI')[METRIC].dropna()

    # Align on common ROIs
    pre_vals  = pre_df.values.astype(float)
    post_vals = post_df.values.astype(float)
    if len(pre_vals) == 0 or len(post_vals) == 0:
        raise ValueError("One period has no usable values.")

    common_rois = list(pre_df.index.union(post_df.index))
    n_pre, n_post = len(pre_vals), len(post_vals)
    n = max(n_pre, n_post)

    t_stat, p_val = stats.ttest_ind(pre_vals, post_vals, equal_var=False)
    summary = {
        'rois':      common_rois,
        'pre_vals':  pre_vals,
        'post_vals': post_vals,
        'pre_mean':  np.mean(pre_vals),
        'post_mean': np.mean(post_vals),
        'pre_sem':   np.std(pre_vals, ddof=1) / np.sqrt(n_pre),
        'post_sem':  np.std(post_vals, ddof=1) / np.sqrt(n_post),
        'n':         n,
        'n_pre':     n_pre,
        'n_post':    n_post,
        't_stat':    t_stat,
        'p_val':     p_val,
    }

    summary['n_responsive'] = len(df['ROI'].unique())

    # --- AUC, aligned on the same ROIs as frequency ---
    if 'mean_auc' in df.columns:
        pre_auc  = df[df['time_period'] == 'pre'].set_index('ROI')['mean_auc']
        post_auc = df[df['time_period'] == 'post'].set_index('ROI')['mean_auc']
        a = pre_auc.dropna().values.astype(float)
        b = post_auc.dropna().values.astype(float)
        n_pre_auc, n_post_auc = len(a), len(b)

        summary['auc'] = {
            'pre_vals':  a,
            'post_vals': b,
            'pre_mean':  np.mean(a) if n_pre_auc else np.nan,
            'post_mean': np.mean(b) if n_post_auc else np.nan,
            'pre_sem':   np.std(a, ddof=1) / np.sqrt(n_pre_auc) if n_pre_auc > 1 else np.nan,
            'post_sem':  np.std(b, ddof=1) / np.sqrt(n_post_auc) if n_post_auc > 1 else np.nan,
            'n':         min(n_pre_auc, n_post_auc),
            'n_pre':     n_pre_auc,
            'n_post':    n_post_auc,
        }

        if n_pre_auc > 1 and n_post_auc > 1:
            t, p = stats.ttest_ind(a, b, equal_var=False)
        else:
            t, p = np.nan, np.nan
        summary['auc']['t_stat'] = t
        summary['auc']['p_val']  = p
    else:
        summary['auc'] = None
   
    summary['panels'] = {m: _one_metric(df, m) for m in PANELS
                         if m in df.columns}
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

def _draw_panel(ax, pre_vals, post_vals, pre_mean, post_mean,
                pre_sem, post_sem, p_val, ylabel):
    """Draw one paired pre/post bar panel with individual ROI points."""
    x      = [0, 1]
    colors = ['#4C8BB5', '#E06B4A']

    ax.bar(x, [pre_mean, post_mean], yerr=[pre_sem, post_sem], width=0.45,
           color=colors, alpha=0.75, capsize=6,
           error_kw=dict(elinewidth=1.5, ecolor='black', capthick=1.5),
           zorder=2)

    for xpos, vals, c in ((0, pre_vals, colors[0]), (1, post_vals, colors[1])):
        j = np.random.uniform(-0.07, 0.07, size=len(vals))
        ax.plot(xpos + j, vals, 'o', color=c, markersize=5, alpha=0.8,
                markeredgecolor='white', markeredgewidth=0.5,
                linestyle='none', zorder=4)

    y_max = max(np.max(pre_vals), np.max(post_vals))
    y_top = y_max * 1.15
    bracket_h = y_max * 0.04
    stars = p_to_stars(p_val)

    ax.plot([0, 0, 1, 1],
            [y_top, y_top + bracket_h, y_top + bracket_h, y_top],
            color='black', linewidth=1.2)
    ax.text(0.5, y_top + bracket_h * 1.3, stars, ha='center', va='bottom',
            fontsize=13, fontweight='bold' if stars != 'ns' else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels(['Pre', 'Post'], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, (y_top + bracket_h) * 1.2)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=10)


PANELS = ['freq_PeaksPerMin', 'mean_amplitude', 'mean_auc']  # edit to taste

def plot_frequency(summary, title='', save_path=None):
    """One paired pre/post panel per metric in PANELS."""
    metrics = [m for m in PANELS if summary['panels'].get(m) is not None]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5.5))
    axes = np.atleast_1d(axes)

    for ax, m in zip(axes, metrics):
        s = summary['panels'][m]
        _draw_panel(ax, s['pre_vals'], s['post_vals'],
                    s['pre_mean'], s['post_mean'],
                    s['pre_sem'], s['post_sem'],
                    s['p_val'], METRIC_LABELS.get(m, m))
        ax.set_title(f'{METRIC_LABELS.get(m, m)}\n'
                     f'pre n={s["n_pre"]}, post n={s["n_post"]}  |  '
                     f'{p_to_stars(s["p_val"])}', fontsize=9)

    subtitle = (f'n = {summary["n_total"]} total ROIs  |  '
                f'{summary["n_with_peaks"]} ROI with peaks  |  Welch t-test')
    fig.suptitle(title + ('\n' if title else '') + subtitle,
                 fontsize=9, color='#555555')
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
    print(f"  Welch t-test  : t = {summary['t_stat']:.3f},  p = {summary['p_val']:.4f}  ({p_to_stars(summary['p_val'])})")
    print(f"{'='*50}\n")

    # Per-ROI table
    print(f"  Pre  n = {summary['n_pre']},  Post n = {summary['n_post']}")
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


    df['ROI'] = df['ROI'].astype(str).str.strip()
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