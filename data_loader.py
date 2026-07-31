import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import percentile_filter


class DataLoader:
    def __init__(self, df, fs):
        self.df = df
        self.fs = fs
        self.n_samples = df.shape[1]

    def get_trace(self, roi_id):
        return self.df.loc[roi_id].values

    def get_times(self):
        return np.arange(self.n_samples) / self.fs


def load_and_preprocess(file_path, file_type, transpose, truncate_samples,
                        smoothing_window_length, poly_order,
                        include_only_cells=False, apply_smoothing=True,
                        baseline_method="mean", baseline_window_samples=None,
                        baseline_percentile=10, subtract_neuropil=False,
                        neuropil_coef=0.7, fs=None):
    """
    Load Suite2p or CSV data and optionally filter to only ROIs where iscell == 1.

    Baseline / ΔF/F options
    -----------------------
    baseline_method : "mean" (default, original behavior) uses a single
        whole-trace mean per ROI as F0. "rolling_percentile" uses a
        time-varying F0 -- a low percentile within a sliding window -- which
        tracks slow drift (photobleaching / z-drift) and reproduces the
        standalone drift-correction script's amplitude scale.
    baseline_window_samples : window length (in samples) for the rolling
        baseline. If None and baseline_method="rolling_percentile", falls back
        to a 60 s window using `fs` (or 1/10 of the trace if fs is unknown).
    baseline_percentile : percentile (0-100) used as the local baseline for the
        rolling method.

    Neuropil subtraction (Suite2p NPY only)
    ---------------------------------------
    subtract_neuropil : if True and the input is a Suite2p .npy, load Fneu.npy
        from the same folder and use (F - neuropil_coef * Fneu) as the trace
        before computing ΔF/F. Ignored for CSV input and if Fneu.npy is absent.
    neuropil_coef : coefficient applied to Fneu (Suite2p default ~0.7).
    fs : sampling rate (Hz), used only to size the rolling baseline window when
        baseline_window_samples is None.
    """

    # -------------------------------
    # Load data
    # -------------------------------
    if file_type.upper() == "CSV":
        df_raw = pd.read_csv(file_path, index_col=0).reset_index(drop=True)
        if 'Err' in df_raw.columns:
            df_raw = df_raw.drop(columns=['Err'])

        # Drop any non-numeric columns (e.g. a 'Control'/'Condition'/'Group'
        # label column). These are not part of the fluorescence time series and
        # would otherwise break the later `data.values.astype(float)` with an
        # error like: could not convert string to float: 'Control'.
        numeric_df = df_raw.apply(pd.to_numeric, errors='coerce')
        non_numeric_cols = [c for c in df_raw.columns
                            if numeric_df[c].isna().all()]
        if non_numeric_cols:
            print(f"[INFO] Dropping non-numeric CSV column(s): "
                  f"{non_numeric_cols}")
            df_raw = df_raw.drop(columns=non_numeric_cols)

        if df_raw.shape[1] == 0:
            raise ValueError(
                "CSV has no numeric data columns after removing label "
                "columns. Check that the file contains fluorescence traces."
            )

        data = df_raw

    else:
        # --- Load the main .npy array (Suite2p fluorescence or spikes) ---
        npy_data = np.load(file_path).astype(float)

        # --- Optional neuropil subtraction: F - coef * Fneu ---
        # Load Fneu.npy from the same folder as F.npy. Only meaningful for the
        # fluorescence trace; silently skipped if Fneu.npy is missing or shapes
        # don't match.
        if subtract_neuropil:
            fneu_path = os.path.join(os.path.dirname(file_path), "Fneu.npy")
            if os.path.exists(fneu_path):
                fneu = np.load(fneu_path).astype(float)
                if fneu.shape == npy_data.shape:
                    npy_data = npy_data - float(neuropil_coef) * fneu
                    print(f"[INFO] Subtracted neuropil (coef={float(neuropil_coef):g}).")
                else:
                    print(f"[WARN] Fneu shape {fneu.shape} != F shape "
                          f"{npy_data.shape}; skipping neuropil subtraction.")
            else:
                print("[WARN] subtract_neuropil=True but no Fneu.npy found — "
                      "skipping neuropil subtraction.")

        data = pd.DataFrame(npy_data)
        data.index = [f"ROI_{i}" for i in range(data.shape[0])]

        # --- Optional Suite2p cell filtering ---
        if include_only_cells:
            folder = os.path.dirname(file_path)
            iscell_path = os.path.join(folder, "iscell.npy")

            if os.path.exists(iscell_path):
                iscell = np.load(iscell_path)
                cell_mask = iscell[:, 0].astype(bool)

                # Apply mask but preserve original Suite2p ROI numbers
                data = data.iloc[cell_mask, :]
                data.index = [f"ROI_{i}" for i in np.where(cell_mask)[0]]
            else:
                print("[WARN] include_only_cells=True but no iscell.npy found — not filtering.")

    # -------------------------------
    # Transpose if necessary
    # -------------------------------
    if transpose:
        data = data.T  # Shape becomes (ROIs, samples)
    
    # -------------------------------
    # Truncate
    # -------------------------------
    total_samples = data.shape[1]  # Always the number of time points
    truncate_samples = max(0, min(truncate_samples, total_samples))  # Clip to valid range
    if truncate_samples > 0:
        data = data.iloc[:, truncate_samples:]
    
    if data.shape[1] < 5:
        raise ValueError(f"Too few samples ({data.shape[1]}) after truncation.")
    

    # -------------------------------
    # Compute ΔF/F
    # -------------------------------
    dff_array = data.values.astype(float)

    if baseline_method == "rolling_percentile":
        # Time-varying F0: a low percentile within a sliding window rides under
        # the transients and tracks slow drift. Reproduces the standalone
        # drift-correction script (and its amplitude scale).
        n_time = dff_array.shape[1]
        win = baseline_window_samples
        if win is None:
            win = int(round(60.0 * fs)) if fs else max(3, n_time // 10)
        win = int(win)
        win = max(1, min(win, n_time))
        if win % 2 == 0:            # percentile_filter is fine with even, but
            win += 1                # keep it odd for symmetry with the script
        pctl = float(baseline_percentile)

        baseline = np.empty_like(dff_array)
        for i in range(dff_array.shape[0]):
            baseline[i] = percentile_filter(dff_array[i], percentile=pctl,
                                            size=win, mode='nearest')
        baseline[baseline == 0] = np.nan
        dff_array = (dff_array - baseline) / baseline
        dff_array = np.nan_to_num(dff_array)
        print(f"[INFO] ΔF/F: rolling {pctl:g}th-pctl baseline, "
              f"window {win} samples"
              + (f" (~{win / fs:.1f} s)." if fs else "."))
    else:
        # Original Calcify behavior: single whole-trace mean baseline per ROI.
        baseline = np.nanmean(dff_array, axis=1, keepdims=True)
        baseline[baseline == 0] = np.nan
        dff_array = (dff_array - baseline) / baseline
        dff_array = np.nan_to_num(dff_array)

    # -------------------------------
    # Smooth (optional)
    # -------------------------------
    if apply_smoothing:
        if data.shape[1] < smoothing_window_length:
            smoothing_window_length = max(3, data.shape[1] | 1)

        if smoothing_window_length < poly_order + 2:
            raise ValueError(
                f"Smoothing window ({smoothing_window_length}) must be >= poly_order + 2 ({poly_order + 2})"
            )

        smoothed_dff = savgol_filter(dff_array, smoothing_window_length, poly_order, axis=1)
    else:
        # No smoothing requested — use the raw ΔF/F as-is.
        smoothed_dff = dff_array

    smoothed_dff_df = pd.DataFrame(smoothed_dff, index=data.index, columns=data.columns)

    return data, smoothed_dff_df
