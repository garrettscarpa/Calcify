import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from PyQt5.QtWidgets import QMessageBox
import os

# np.trapz was renamed to np.trapezoid in NumPy 2.0 and removed under the old
# name. Use whichever exists so the code runs on NumPy 1.x and 2.x alike.
_trapz = getattr(np, "trapezoid", None) or np.trapz

class PeakDetector:
    def __init__(self, smoothed_dff_df, fs, prominence, min_height=None, min_plateau_samples=None):
        self.smoothed_dff_df = smoothed_dff_df
        self.fs = fs
        self.prominence = prominence
        self.min_height = min_height
        self.min_plateau_samples = min_plateau_samples
        self.peak_results_df = self._detect_peaks()

    @staticmethod
    def _find_left_base(trace, peak_idx):
        # Search left from peak to start of trace
        local_min_idx = None
        min_val = trace[peak_idx]
        for i in range(peak_idx - 1, 0, -1):  # From peak-1 down to 1
            if trace[i] < trace[i - 1] and trace[i] < trace[i + 1]:
                return i
            if trace[i] < min_val:
                min_val = trace[i]
                local_min_idx = i
        return local_min_idx if local_min_idx is not None else 0

    @staticmethod
    def _find_right_base(trace, peak_idx):
        # Search right from peak to end of trace
        local_min_idx = None
        min_val = trace[peak_idx]
        for i in range(peak_idx + 1, len(trace) - 1):
            if trace[i] < trace[i - 1] and trace[i] < trace[i + 1]:
                return i
            if trace[i] < min_val:
                min_val = trace[i]
                local_min_idx = i
        return local_min_idx if local_min_idx is not None else len(trace) - 1

    def _detect_peaks(self):
        all_peaks = []
        for cell_id, trace in self.smoothed_dff_df.iterrows():
            trace_values = trace.values
            peaks, props = find_peaks(trace_values, prominence=self.prominence)
            
            for i, peak_idx in enumerate(peaks):
                prominence_value = props['prominences'][i]
            
                # Redundant but explicit: enforce prominence threshold
                if prominence_value < self.prominence:
                    continue
            
                left_base = props['left_bases'][i]
                right_base = props['right_bases'][i]
                
                if right_base < left_base:
                    right_base = left_base
            
                base_value = min(trace_values[left_base], trace_values[right_base])
                peak_value = trace_values[peak_idx]
            
                # Apply min_height filter
                if self.min_height is not None and peak_value < self.min_height:
                    continue
            
                # Apply plateau length filter
                if self.min_plateau_samples is not None:
                    plateau_length = right_base - left_base
                    if plateau_length < self.min_plateau_samples:
                        continue

    
                half_max = base_value + 0.5 * (peak_value - base_value)
                time_to_peak = (peak_idx - left_base) / self.fs
    
                # Calculate half-rise time
                half_rise_time = np.nan
                rise_segment = trace_values[left_base:peak_idx + 1]
                rise_indices = np.arange(left_base, peak_idx + 1)
                above_half = np.where(rise_segment >= half_max)[0]
                if len(above_half) > 0 and above_half[0] > 0:
                    j = above_half[0]
                    x0, x1 = rise_indices[j - 1], rise_indices[j]
                    y0, y1 = trace_values[x0], trace_values[x1]
                    if y1 != y0:
                        frac = (half_max - y0) / (y1 - y0)
                        half_rise_time = (x0 + frac - left_base) / self.fs
    
                # Calculate half-decay time
                half_decay_time = np.nan
                decay_segment = trace_values[peak_idx:right_base + 1]
                decay_indices = np.arange(peak_idx, right_base + 1)
                below_half = np.where(decay_segment <= half_max)[0]
                if len(below_half) > 0 and below_half[0] > 0:
                    j = below_half[0]
                    x0, x1 = decay_indices[j - 1], decay_indices[j]
                    y0, y1 = trace_values[x0], trace_values[x1]
                    if y1 != y0:
                        frac = (half_max - y0) / (y1 - y0)
                        half_decay_time = (x0 + frac - peak_idx) / self.fs
    
                time_segment = np.arange(left_base, right_base + 1) / self.fs
                peak_segment = np.maximum(trace_values[left_base:right_base + 1], base_value)
                auc = _trapz(np.abs(peak_segment - base_value), time_segment)
                prominence_value = props['prominences'][i]
    
                peak_info = {
                    'cell_id': cell_id,
                    'peak_time': peak_idx / self.fs,
                    'left_bases': left_base / self.fs,
                    'right_bases': right_base / self.fs,
                    'prominences': prominence_value,
                    'base_value': base_value,
                    'peak_value': peak_value,
                    'auc': auc,
                    'time_to_peak': time_to_peak,
                    'half_rise_time': half_rise_time,
                    'half_decay_time': half_decay_time
                }
                all_peaks.append(peak_info)
    
        return pd.DataFrame(all_peaks).sort_values(['cell_id', 'peak_time'])


    def get_peak_dataframe(self):
        return self.peak_results_df

def load_or_detect_peaks(app, file_path, smoothed_dff_df, fs,
                         prominence=None, min_height=None, min_plateau_samples=None):
    """
    Load existing filtered peaks if available, otherwise detect peaks.
    
    Parameters
    ----------
    app : QWidget
        Main app window (for QMessageBox warnings).
    file_path : str
        Path to the original data file (NPY/CSV). Determines where to look for F_filtered_peaks.csv.
    smoothed_dff_df : pd.DataFrame
        DataFrame of smoothed ΔF/F values (ROIs x timepoints).
    fs : float
        Sampling rate in Hz.
    prominence : float, optional
        Minimum prominence for peak detection.
    min_height : float, optional
        Minimum peak height for detection.
    min_plateau_samples : int, optional
        Minimum plateau width for detection.
    
    Returns
    -------
    pd.DataFrame or None
        Peak results with columns ['cell_id', 'peak_time', ...] or None if no peaks found.
    """
    # Determine export path for filtered peaks
    export_dir = os.path.dirname(file_path)
    file_prefix = os.path.splitext(os.path.basename(file_path))[0] \
        if file_path.endswith(".csv") \
        else os.path.basename(os.path.dirname(file_path))
    
    filtered_peaks_path = os.path.join(export_dir, f"{file_prefix}_filtered_peaks.csv")
    
    # 1) Load CSV if it exists
    if os.path.isfile(filtered_peaks_path):
        try:
            peak_results_df = pd.read_csv(filtered_peaks_path)
        except Exception as e:
            QMessageBox.warning(app, "Error", f"Could not load saved peaks:\n{e}")
            return None

        if peak_results_df.empty:
            QMessageBox.warning(app, "No Peaks Found", "Saved peak file contains no peaks.")
            return None

        # Ensure consistent ROI strings
        if "cell_id" in peak_results_df.columns:
            peak_results_df["cell_id"] = peak_results_df["cell_id"].astype(str)

        # Warn if CSV ROIs don't match current data
        missing_rois = set(peak_results_df["cell_id"]) - set(smoothed_dff_df.index)
        if missing_rois:
            print(f"Warning: these ROIs from CSV are missing in current data: {missing_rois}")

        # Keep only ROIs present in current data
        peak_results_df = peak_results_df[peak_results_df["cell_id"].isin(smoothed_dff_df.index)]

        # Sort by cell_id and peak_time
        peak_results_df = peak_results_df.sort_values(["cell_id", "peak_time"])

        return peak_results_df

    # 2) Otherwise, detect peaks
    from peak_analysis import PeakDetector

    if prominence is None:
        QMessageBox.warning(app, "Missing Prominence",
                            "A prominence value is required for peak detection.")
        return None

    detector = PeakDetector(
        smoothed_dff_df,
        fs,
        prominence=prominence,
        min_height=min_height,
        min_plateau_samples=min_plateau_samples
    )

    peak_results_df = detector.get_peak_dataframe()

    if peak_results_df.empty:
        QMessageBox.warning(app, "No Peaks Detected",
                            "No peaks were detected with the current parameters.")
        return None

    return peak_results_df