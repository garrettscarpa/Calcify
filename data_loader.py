import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


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
                        include_only_cells=False, apply_smoothing=True):
    """
    Load Suite2p or CSV data and optionally filter to only ROIs where iscell == 1.
    """

    # -------------------------------
    # Load data
    # -------------------------------
    if file_type.upper() == "CSV":
        df_raw = pd.read_csv(file_path, index_col=0).reset_index(drop=True)
        if 'Err' in df_raw.columns:
            df_raw = df_raw.drop(columns=['Err'])
        data = df_raw

    else:
        # --- Load the main .npy array (Suite2p fluorescence or spikes) ---
        npy_data = np.load(file_path)
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
    dff_array = data.values
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
