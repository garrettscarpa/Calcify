from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QVBoxLayout, QPushButton, QCheckBox
from data_loader import DataLoader
from ui_setup import build_ui
from file_handler import filetype_changed, select_file_or_folder, populate_file_list
from data_loader import load_and_preprocess
from peak_analysis import load_or_detect_peaks
from plotting import setup_plot_and_interaction
from scipy.interpolate import CubicSpline
import os
import pandas as pd
from matplotlib.widgets import SpanSelector
from scipy.signal import butter, filtfilt
cutoff_hz = 0.05
z_thresh = 1.5
min_len = 1
context = 10
artifact_removal_padding = 2  # Number of frames to expand each artifact region by



def mask_to_regions(mask, min_len=min_len):
    """
    Convert boolean mask into contiguous index regions.
    """
    regions = []
    in_region = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_region:
            start = i
            in_region = True
        elif not val and in_region:
            end = i - 1
            if end - start + 1 >= min_len:
                regions.append((start, end))
            in_region = False

    if in_region:
        end = len(mask) - 1
        if end - start + 1 >= min_len:
            regions.append((start, end))

    return regions


def merge_regions(regions, gap=0):
    """
    Merge overlapping or near-adjacent (start,end) index regions into
    non-overlapping ones. `gap` lets regions separated by <= gap samples
    be joined too. Without this, removing overlapping regions one-by-one
    makes each interpolation anchor on the previous one's flattened edge,
    corrupting the result.
    """
    if not regions:
        return []
    regions = sorted(regions)
    merged = [list(regions[0])]
    for s, e in regions[1:]:
        if s <= merged[-1][1] + 1 + gap:        # Overlapping or touching
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

def highpass_filter(signal, fs, cutoff_hz, order=3):
    """
    Zero-phase Butterworth high-pass filter.
    """
    nyq = 0.5 * fs
    norm_cutoff = cutoff_hz / nyq
    b, a = butter(order, norm_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)


class CalciumImagingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calcium Imaging Analysis")
        self.csv_folder_path = None
        self.npy_folder_path = None
        self.figures = []

        layout, ui = build_ui(self)
        self.setLayout(layout)

        # Widgets
        self.file_list_widget = ui['file_list_widget']
        self.filetype_combo = ui['filetype_combo']
        self.file_button = ui['file_button']
        self.file_label = ui['file_label']
        self.npy_filename_input = ui['npy_filename_input']
        self.fs_input = ui['fs_input']
        self.truncate_seconds_input = ui['truncate_seconds_input']
        self.prominence_input = ui['prominence_input']
        self.min_height_input = ui['min_height_input']
        self.min_plateau_input = ui['min_plateau_input']
        self.poly_order_input = ui['poly_order_input']
        self.smoothing_window_input = ui['smoothing_window_input']
        self.savgol_checkbox = ui['savgol_checkbox']
        self.display_window_input = ui['display_window_input']
        self.yax_input = ui['yax_input']
        self.transpose_checkbox = ui['transpose_checkbox']
        self.pre_range_input = ui['pre_range_input']
        self.post_range_input = ui['post_range_input']
        self.run_button = ui['run_button']
        self.plot_all_button = ui['plot_all_button']
        self.cell_filter_checkbox = ui['cell_filter_checkbox']

        # Signals
        self.filetype_combo.currentIndexChanged.connect(lambda: filetype_changed(self))
        self.file_button.clicked.connect(lambda: select_file_or_folder(self))
        self.npy_filename_input.textChanged.connect(lambda: populate_file_list(self))
        self.run_button.clicked.connect(self.run_analysis)
        self.plot_all_button.clicked.connect(self.plot_all_rois)
        # Auto-fill fs from Suite2p ops.npy whenever the selected file changes.
        self.file_list_widget.itemSelectionChanged.connect(self.autofill_fs_from_ops)
        self.loaded_file = None
        self._last_load_settings = None

    # ------------------------------------------------------------------
    # Sampling-rate auto-detection (Suite2p ops.npy)
    # ------------------------------------------------------------------
    @staticmethod
    def detect_fs_from_ops(selected_file):
        """
        Return the acquisition sampling rate (Hz) for a Suite2p recording by
        reading 'fs' from ops.npy in the same folder as the selected file.
        Returns None if it can't be determined (e.g. CSV input, no ops.npy).
        """
        if not selected_file or not selected_file.lower().endswith('.npy'):
            return None
        ops_path = os.path.join(os.path.dirname(selected_file), 'ops.npy')
        if not os.path.exists(ops_path):
            return None
        try:
            ops = np.load(ops_path, allow_pickle=True).item()
            fs = ops.get('fs', None)
            return float(fs) if fs is not None else None
        except Exception as e:
            print(f"[autofill_fs] Could not read fs from ops.npy: {e}")
            return None

    def autofill_fs_from_ops(self):
        """When a Suite2p .npy is selected, set the fs box to ops.npy's fs."""
        items = self.file_list_widget.selectedItems()
        if not items:
            return
        detected = self.detect_fs_from_ops(items[0].text())
        if detected is not None:
            self.fs_input.setText(f"{detected:.6g}")
            print(f"[autofill_fs] Detected fs = {detected:.6g} Hz from ops.npy")
        
    def clear_artifact_spans(self, ax):
        """
        Remove previously drawn artifact spans without touching other patches.
        """
        to_remove = [
            p for p in ax.patches
            if getattr(p, "_is_artifact_span", False)
        ]
        for p in to_remove:
            p.remove()

    def apply_truncation(self, df, fs):
        truncate_seconds = float(self.truncate_seconds_input.text())
        truncate_samples = int(truncate_seconds * fs)
    
        if truncate_samples > 0:
            return df.iloc[:, truncate_samples:]
        return df

    def adjust_artifact_regions_for_truncation(self, truncate_samples):
        """
        Shift artifact regions after truncation.
        Removes regions that fall entirely before the truncation point.
        """
        if not hasattr(self, "interpolated_regions"):
            return
    
        adjusted = []
    
        for start, end in self.interpolated_regions:
            # Region falls entirely before the truncation point: drop it
            if end < truncate_samples:
                continue
    
            # Shift and clamp
            new_start = max(0, start - truncate_samples)
            new_end   = max(0, end - truncate_samples)
    
            if new_end > new_start:
                adjusted.append((new_start, new_end))
    
        self.interpolated_regions = adjusted
    

    def auto_remove_motion_artifacts(
        self,
        z_thresh=z_thresh,
        min_len=min_len,
        context=context,
        use_std=False
    ):
        """
        Automatically detect and interpolate motion artifacts
        using population-wide signal deviations.
        """
    
        if not hasattr(self, "current_data"):
            QMessageBox.warning(self, "Error", "No data loaded.")
            return
    
        df = self.current_data
    
        # --- Global population signal ---
        global_signal = (
            df.std(axis=0).values if use_std
            else df.mean(axis=0).values
        )
        
        # --- High-pass filter to remove slow drift ---
        fs = float(self.fs_input.text())
        global_signal = highpass_filter(
            global_signal,
            fs=fs,
            cutoff_hz=cutoff_hz  # Sensible default (~30 s period)
        )

        # --- Robust z-score ---
        mu = np.median(global_signal)
        sigma = np.std(global_signal)
    
        if sigma == 0:
            QMessageBox.warning(self, "Error", "Global signal variance is zero.")
            return
    
        z = (global_signal - mu) / sigma
        artifact_mask = np.abs(z) >= z_thresh
    
        # --- Convert mask to contiguous regions ---
        regions = mask_to_regions(artifact_mask, min_len=min_len)
    
        if not regions:
            QMessageBox.information(
                self, "Auto Artifact Removal",
                "No motion artifacts detected."
            )
            return
            
        # --- Pad, then MERGE overlapping/adjacent regions ---
        # Padding each region can make neighbours overlap; removing overlapping
        # regions sequentially corrupts the interpolation (each anchors on the
        # previous fill). Merge into clean, non-overlapping spans first, then
        # remove each exactly once.
        n = df.shape[1]
        padded = [
            (max(0, s - artifact_removal_padding),
             min(n - 1, e + artifact_removal_padding))
            for s, e in regions
        ]
        merged = merge_regions(padded, gap=context)

        for start_idx, end_idx in merged:
            self.remove_artifact(start_idx, end_idx, context=context)

        print(f"Auto-removed {len(merged)} motion artifact region(s) "
              f"(merged from {len(regions)} detected).")

    def attach_artifact_redraw(self, fig, fs):
        """
        Ensure artifact shading is reapplied whenever the figure redraws
        (e.g., peak navigation).
        """
        if not fig.axes:
            return
    
        ax0 = fig.axes[0]
    
        def _on_draw(event):
            # Avoid infinite recursion
            if not hasattr(self, "interpolated_regions"):
                return
    
            # Reapply shading only if spans are missing
            existing_spans = [
                p for p in ax0.patches
                if hasattr(p, "get_facecolor")
            ]
    
            if not existing_spans:
                self.highlight_interpolated_regions(ax0, fs)
    
        fig.canvas.mpl_connect("draw_event", _on_draw)

    # ------------------------------------------------------------------
    # Artifact-region persistence and highlighting
    # ------------------------------------------------------------------
    def save_artifact_regions(self, fs, export_dir, file_prefix):
        if not hasattr(self, "interpolated_regions"):
            return
    
        rows = []
        for start_idx, end_idx in self.interpolated_regions:
            rows.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time_s": start_idx / fs,
                "end_time_s": end_idx / fs,
            })
    
        df = pd.DataFrame(rows)
        csv_path = os.path.join(export_dir, f"{file_prefix}_artifact_regions.csv")
        df.to_csv(csv_path, index=False)
    
        print(f"Artifact regions saved to {csv_path}")



    def load_artifact_regions(self, fs, export_dir, file_prefix):
        csv_path = os.path.join(export_dir, f"{file_prefix}_artifact_regions.csv")
    
        if not os.path.exists(csv_path):
            return
    
        df = pd.read_csv(csv_path)
    
        self.interpolated_regions = [
            (int(r.start_idx), int(r.end_idx))
            for r in df.itertuples()
        ]
    
        print(f"Loaded {len(self.interpolated_regions)} artifact regions")
        
    def highlight_interpolated_regions(self, ax, fs):
        if not hasattr(self, 'interpolated_regions'):
            return
    
        # Clear any previously drawn spans before redrawing
        self.clear_artifact_spans(ax)

        for region_start, region_end in self.interpolated_regions:
            t_start = region_start / fs
            t_end   = region_end / fs
            span = ax.axvspan(t_start, t_end, color='red', alpha=0.3)

            # Tag so the span can be identified and removed later
            span._is_artifact_span = True

    
    # ------------------------------------------------------------------
    # Plot All ROIs window
    # ------------------------------------------------------------------
    def plot_all_rois(self):
        try:
            fs = float(self.fs_input.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid sampling rate (fs).")
            return
    
        selected_items = self.file_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No selection", "Please select one .npy recording.")
            return
    
        selected_file = selected_items[0].text()
        file_prefix = os.path.splitext(selected_file)[0]
        export_dir = os.path.dirname(selected_file)
        interpolated_file = os.path.join(export_dir, f"{file_prefix}_interpolated.npy")
        
        transpose = self.transpose_checkbox.isChecked()
        cell_filter = self.cell_filter_checkbox.isChecked()
        load_settings = (selected_file, transpose, cell_filter)
        
        if self._last_load_settings != load_settings:
            self.current_data = None          # Force a fresh load
            self._last_load_settings = load_settings
    
        # Load interpolated file if it exists, otherwise preprocess
        if os.path.exists(interpolated_file):
            print(f"Loading interpolated data from {interpolated_file}")
            smoothed_dff_df = pd.DataFrame(
                np.array(np.load(interpolated_file, allow_pickle=True), dtype=float, copy=True),
                index=getattr(self, 'original_index', None),
                columns=getattr(self, 'original_columns', None)
            )
            truncate_seconds = float(self.truncate_seconds_input.text())
            truncate_samples = int(truncate_seconds * fs)
            
            smoothed_dff_df = self.apply_truncation(smoothed_dff_df, fs)
            self.current_data = smoothed_dff_df.copy()
            
            self.load_artifact_regions(fs, export_dir, file_prefix)
            self.adjust_artifact_regions_for_truncation(truncate_samples)

        elif self.current_data is not None:
            smoothed_dff_df = self.current_data.copy()
        else:
            try:
                truncate_seconds = float(self.truncate_seconds_input.text())
                truncate_samples = int(truncate_seconds * fs)
                transpose = self.transpose_checkbox.isChecked()
                smoothing_window_length = int(self.smoothing_window_input.text())
                poly_order = int(self.poly_order_input.text())
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid numeric value in fields.")
                return
    
            data, smoothed_dff_df = load_and_preprocess(
                selected_file,
                self.filetype_combo.currentText(),
                transpose,
                truncate_samples,
                smoothing_window_length,
                poly_order,
                include_only_cells=self.cell_filter_checkbox.isChecked(),
                apply_smoothing=self.savgol_checkbox.isChecked()
            )
    
            self.original_index = smoothed_dff_df.index
            self.original_columns = smoothed_dff_df.columns
            self.current_data = smoothed_dff_df.copy()
    
        # Create plot window
        self.plot_window = QWidget()
        self.plot_window.setWindowTitle("Plot All ROIs")
        layout = QVBoxLayout(self.plot_window)
    
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
    
        self.remove_region_button = QPushButton("Remove Selected Region")
        self.remove_region_button.clicked.connect(self.remove_selected_region)
        layout.addWidget(self.remove_region_button)
    
        self.auto_remove_button = QPushButton("Auto Remove Motion Artifacts")
        self.auto_remove_button.clicked.connect(
            lambda: self.auto_remove_motion_artifacts(
                z_thresh=z_thresh,   # Adjustable threshold
                min_len=min_len,
                context=context,
                use_std=False   # Set True to use population std instead of mean
            )
        )
        layout.addWidget(self.auto_remove_button)

        # Optional average-trace overlay. Off by default (current behavior);
        # when checked, plots the mean across all ROIs on top of the traces.
        self.avg_trace_checkbox = QCheckBox("Show average trace")
        self.avg_trace_checkbox.setChecked(False)
        self.avg_trace_checkbox.toggled.connect(self.toggle_average_trace)
        layout.addWidget(self.avg_trace_checkbox)

        self.ax = self.fig.add_subplot(111)
        
        # Set axis labels
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel(r"$\Delta f/f$")
        
        n_samples = self.current_data.shape[1]
        t = np.arange(n_samples) / fs
        self.roi_lines = {}
        
        for roi in self.current_data.index:
            line, = self.ax.plot(
                t,
                self.current_data.loc[roi],
                linewidth=1,
                alpha=0.8
            )
            self.roi_lines[roi] = line

        # Keep fs and time vector available for the average-trace toggle.
        self._plot_all_fs = fs
        self._plot_all_time = t
        self.avg_trace_line = None
        # Respect the checkbox state (default off, so normally a no-op).
        if self.avg_trace_checkbox.isChecked():
            self._draw_average_trace()

       
        self.span_selector = SpanSelector(
            self.ax,
            self.on_select_span,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='yellow')
        )

        y_min = self.current_data.min().min()
        y_max = self.current_data.max().max()
        padding = 0.05 * (y_max - y_min)
        self.ax.set_ylim(y_min - padding, y_max + padding)

        # Highlight any previously interpolated regions
        self.highlight_interpolated_regions(self.ax, fs)

    
        self.canvas.draw_idle()
        self.plot_window.show()

    # ------------------------------------------------------------------
    # Average-trace overlay
    # ------------------------------------------------------------------
    def _draw_average_trace(self):
        """Plot the mean trace across all ROIs as a thick black line."""
        if not hasattr(self, "current_data") or self.current_data is None:
            return
        t = getattr(self, "_plot_all_time", None)
        if t is None:
            t = np.arange(self.current_data.shape[1]) / self._plot_all_fs

        mean_trace = self.current_data.mean(axis=0).values
        if self.avg_trace_line is not None:
            # Update existing line in place.
            self.avg_trace_line.set_data(t, mean_trace)
        else:
            self.avg_trace_line, = self.ax.plot(
                t, mean_trace,
                color='black', linewidth=2.5, alpha=0.9,
                zorder=10, label='Average'
            )
        self.canvas.draw_idle()

    def toggle_average_trace(self, checked):
        """Show or hide the average-trace overlay based on the checkbox."""
        if checked:
            self._draw_average_trace()
        else:
            if getattr(self, "avg_trace_line", None) is not None:
                self.avg_trace_line.remove()
                self.avg_trace_line = None
                self.canvas.draw_idle()


    def on_select_span(self, xmin, xmax):
        print("Span selected (stored only):", xmin, xmax)
        

        # Store temporarily
        self.pending_xmin = xmin
        self.pending_xmax = xmax

        # Remove old yellow selection if present
        if hasattr(self, "pending_span_patch") and self.pending_span_patch in self.ax.patches:
            self.pending_span_patch.remove()

        # Draw temporary yellow highlight
        self.pending_span_patch = self.ax.axvspan(
            xmin, xmax, color='yellow', alpha=0.3
        )

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Remove selected region
    # ------------------------------------------------------------------
    def remove_selected_region(self):

        if not hasattr(self, "pending_xmin"):
            QMessageBox.warning(self, "Error", "No region selected.")
            return

        xmin = self.pending_xmin
        xmax = self.pending_xmax

        fs = float(self.fs_input.text())
        n_samples = self.current_data.shape[1]

        start_idx = int(np.clip(xmin * fs, 0, n_samples - 1))
        end_idx   = int(np.clip(xmax * fs, 0, n_samples - 1))

        # Remove & replot
        self.remove_artifact(start_idx, end_idx)

        # Clear temporary selection
        del self.pending_xmin
        del self.pending_xmax
            
    def remove_artifact(self, start_idx, end_idx, context=2):
        """
        Remove an artifact region by drawing a straight line between the edges.
        Automatically updates current_data and saves the interpolated .npy file.
        Works for both manual selection and auto-detected artifacts.
        """
    
        # Deep copy with a guaranteed-writable backing array (np.load can hand
        # back read-only buffers that propagate through .copy()).
        df = self.current_data.copy()
        df = pd.DataFrame(
            np.array(df.values, dtype=float, copy=True),
            index=df.index, columns=df.columns
        )
        n_samples = df.shape[1]
    
        if not hasattr(self, 'interpolated_regions'):
            self.interpolated_regions = []
        self.interpolated_regions.append((start_idx, end_idx))
        fs = float(self.fs_input.text())
        selected_items = self.file_list_widget.selectedItems()
        if selected_items:
            selected_file = selected_items[0].text()
            file_prefix = os.path.splitext(selected_file)[0]
            export_dir = os.path.dirname(selected_file)
        
            self.save_artifact_regions(fs, export_dir, file_prefix)

    
        for roi in df.index:
            # .values may return a read-only buffer when the frame is backed
            # by an mmap'd / np.load'd array (the interpolated .npy case). Take
            # an explicit writable copy; otherwise in-place assignment raises
            # "assignment destination is read-only", which is why auto-remove
            # (run after an interpolated file was reloaded) silently failed.
            y = np.array(df.loc[roi].values, dtype=float, copy=True)

            # Use context frames to define edges
            before_start = max(0, start_idx - context)
            after_end = min(n_samples - 1, end_idx + context)

            y0 = y[before_start]
            y1 = y[after_end]

            # Linear interpolation across artifact
            n_points = end_idx - start_idx + 1
            y[start_idx:end_idx+1] = np.linspace(y0, y1, n_points)

            df.loc[roi] = y
    
        self.current_data = df
    
        # Auto-save interpolated file
        if hasattr(self, 'file_list_widget'):
            selected_items = self.file_list_widget.selectedItems()
            if selected_items:
                selected_file = selected_items[0].text()
                file_prefix = os.path.splitext(selected_file)[0]
                export_dir = os.path.dirname(selected_file)
                interpolated_file = os.path.join(export_dir, f"{file_prefix}_interpolated.npy")
                np.save(interpolated_file, self.current_data.values)
                print(f"Interpolated data saved to {interpolated_file}")
    
        # Replot updated data
        fs = float(self.fs_input.text())
        t = np.arange(self.current_data.shape[1]) / fs
    

        # Update existing line data only
        for roi in self.current_data.index:
            self.roi_lines[roi].set_ydata(self.current_data.loc[roi])

        # If the average overlay is currently shown, recompute it on the
        # newly interpolated data so it stays in sync with the ROI traces.
        if getattr(self, "avg_trace_line", None) is not None:
            self._draw_average_trace()
        
        # Update artifact spans
        self.highlight_interpolated_regions(self.ax, fs)
        
        # One safe redraw
        self.canvas.draw_idle()
            
        
    def run_analysis(self):
        
        # --- Check file selection ---
        selected_items = self.file_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No selection", "Please select one .npy recording from the list.")
            return
    
        selected_file = selected_items[0].text()
        file_type = self.filetype_combo.currentText()
        valid_ext = '.csv' if file_type == 'CSV' else '.npy'
        if not selected_file.lower().endswith(valid_ext):
            QMessageBox.warning(self, "Invalid file type", f"Please select a {valid_ext} file.")
            return
        transpose = self.transpose_checkbox.isChecked()
        cell_filter = self.cell_filter_checkbox.isChecked()
        load_settings = (selected_file, transpose, cell_filter)
        
        if self._last_load_settings != load_settings:
            self.current_data = None          # Force a fresh load
            self._last_load_settings = load_settings
                
        
        # --- Reset analysis state if the selected file changed ---
        if self.loaded_file != selected_file:
            print("[INFO] New file selected — resetting analysis state")
        
            self.current_data = None
            self.original_index = None
            self.original_columns = None
            self.interpolated_regions = []
        
            # Close old figures cleanly
            for fig in self.figures:
                plt.close(fig)
            self.figures.clear()
        
            self.loaded_file = selected_file



        # --- Basic numeric parameters ---
        try:
            fs = float(self.fs_input.text())
            truncate_seconds = float(self.truncate_seconds_input.text())
            truncate_samples = int(truncate_seconds * fs)
            transpose = self.transpose_checkbox.isChecked()
            smoothing_window_length = int(self.smoothing_window_input.text())
            poly_order = int(self.poly_order_input.text())
            y_ax_range = tuple(map(float, self.yax_input.text().split(',')))
            display_window = float(self.display_window_input.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid numeric values.")
            return

        # --- Sampling-rate sanity check against Suite2p ops.npy ---
        # A wrong fs silently corrupts every time<->index conversion (this is the
        # classic "peaks land outside the trace" bug). Warn before proceeding.
        detected_fs = self.detect_fs_from_ops(selected_file)
        if detected_fs is not None and abs(detected_fs - fs) > 1e-3:
            choice = QMessageBox.warning(
                self, "Sampling rate mismatch",
                f"The fs you entered ({fs:g} Hz) does not match the rate stored "
                f"in ops.npy ({detected_fs:.6g} Hz).\n\n"
                f"Using the wrong fs will misplace every peak in time.\n\n"
                f"Use the detected value ({detected_fs:.6g} Hz)?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if choice == QMessageBox.Yes:
                fs = detected_fs
                self.fs_input.setText(f"{detected_fs:.6g}")
                truncate_samples = int(truncate_seconds * fs)
                print(f"[run_analysis] Using detected fs = {fs:.6g} Hz")
            else:
                print(f"[run_analysis] Proceeding with user fs = {fs:g} Hz "
                      f"(ops.npy says {detected_fs:.6g})")

        # --- Optional peak detection parameters ---
        peak_params = {}
        if self.prominence_input.text():
            peak_params['prominence'] = float(self.prominence_input.text())
        if self.min_height_input.text():
            peak_params['min_height'] = float(self.min_height_input.text())
        if self.min_plateau_input.text():
            peak_params['min_plateau_samples'] = int(self.min_plateau_input.text())
    
        # --- Load interpolated or current data ---
        # Prefix used for sidecar files. For NPY/Suite2p the meaningful name is
        # the recording folder (e.g. "F" lives in .../plane0/), so derive the
        # prefix the same way peak_analysis.load_or_detect_peaks does, and keep
        # it as a bare name (not a full path) to avoid stale-path mismatches.
        if selected_file.lower().endswith('.csv'):
            file_prefix = os.path.splitext(os.path.basename(selected_file))[0]
        else:
            file_prefix = os.path.splitext(os.path.basename(selected_file))[0]
        export_dir = os.path.dirname(selected_file)
        interpolated_file = os.path.join(export_dir, f"{file_prefix}_interpolated.npy")
        
        if os.path.exists(interpolated_file):
            print(f"Loading interpolated data from {interpolated_file}")
            smoothed_dff_df = pd.DataFrame(
                np.array(np.load(interpolated_file, allow_pickle=True), dtype=float, copy=True),
                index=getattr(self, 'original_index', None),
                columns=getattr(self, 'original_columns', None)
            )
            self.load_artifact_regions(fs, export_dir, file_prefix)
        
        elif self.current_data is not None:
            smoothed_dff_df = self.current_data.copy()

        
        else:
            data, smoothed_dff_df = load_and_preprocess(
                selected_file,
                self.filetype_combo.currentText(),
                transpose,
                truncate_samples,
                smoothing_window_length,
                poly_order,
                include_only_cells=self.cell_filter_checkbox.isChecked(),
                apply_smoothing=self.savgol_checkbox.isChecked()
            )
            self.original_index = smoothed_dff_df.index
            self.original_columns = smoothed_dff_df.columns
            self.load_artifact_regions(fs, export_dir, file_prefix)
        
            # If artifact removal was already applied in Plot All ROIs,
            # use that interpolated data instead of the freshly loaded version
            if self.current_data is not None:
                smoothed_dff_df = self.current_data.copy()
        
        # ------------------------------------------------
        # Apply truncation and realignment (critical)
        # ------------------------------------------------
        smoothed_dff_df = self.apply_truncation(smoothed_dff_df, fs)
        self.current_data = smoothed_dff_df.copy()
        self.adjust_artifact_regions_for_truncation(truncate_samples)
        

        # --- Find filtered peaks CSV dynamically ---
        # Prefer the plain "<prefix>_filtered_peaks.csv"; never pick the
        # "_timelocked" variant, which uses a different (experiment) timebase
        # and would place peaks outside the loaded trace. This matches the
        # exclusion logic in file_handler.py and generate_plots.py.
        filtered_csv = None
        candidates = [
            f for f in os.listdir(export_dir)
            if 'filtered_peaks' in f and f.endswith('.csv')
            and 'timelocked' not in f
        ]
        if candidates:
            # Prefer one whose prefix matches this recording, else first.
            base = os.path.basename(file_prefix)
            preferred = [c for c in candidates if base and base in c]
            chosen = preferred[0] if preferred else candidates[0]
            filtered_csv = os.path.join(export_dir, chosen)

        if filtered_csv is not None and os.path.exists(filtered_csv):
            print(f"Loading filtered peaks from: {filtered_csv}")
            peak_results_df = pd.read_csv(filtered_csv)
            csv_times = peak_results_df['peak_time'].values[:5]
            print("First 5 peak_time from CSV:", csv_times)
        else:
            # Fall back to dynamic peak detection
            peak_results_df = load_or_detect_peaks(
                app=self,
                file_path=selected_file,
                smoothed_dff_df=smoothed_dff_df,
                fs=fs,
                **peak_params
            )
        # ------------------------------------------------
        # Initialize base-related columns (required downstream)
        # ------------------------------------------------
        for col in [
            'left_bases', 'right_bases',
            'left_base_value', 'right_base_value',
            'base_value'
        ]:
            if col not in peak_results_df.columns:
                peak_results_df[col] = np.nan
            peak_results_df[col] = peak_results_df[col].astype(float)

        # --- Handle peak indices ---
        peak_results_df['peak_idx'] = (peak_results_df['peak_time'] * fs).astype(int)
        print("\n>>> PEAK INDEX CHECK")
        print("Total peaks:", len(peak_results_df))
        print("Min peak_idx:", peak_results_df['peak_idx'].min())
        print("Max peak_idx:", peak_results_df['peak_idx'].max())
        print("Data length:", smoothed_dff_df.shape[1])

        # Ensure 'cell_id' exists
        if 'cell_id' not in peak_results_df.columns:
            if 'ROI' in peak_results_df.columns:
                peak_results_df['cell_id'] = peak_results_df['ROI']
            else:
                raise KeyError("Peak DataFrame must have either 'cell_id' or 'ROI' column.")
    
        # --- Clean up CSV cell IDs ---
        peak_results_df['cell_id'] = (
            peak_results_df['cell_id'].astype(str)
            .str.strip()
            .str.replace('cell_', '', regex=False)
            .str.replace('ROI_', '', regex=False)
        )
    
        # --- Clean up smoothed_dff_df index to match CSV ---
        smoothed_dff_df.index = smoothed_dff_df.index.astype(str).str.strip()
        smoothed_dff_df.index = smoothed_dff_df.index.str.replace('cell_', '', regex=False)
        smoothed_dff_df.index = smoothed_dff_df.index.str.replace('ROI_', '', regex=False)
    
        # --- Filter peaks to only valid ROIs present in smoothed_dff_df ---
        valid_rois = smoothed_dff_df.index
        peak_results_df = peak_results_df[peak_results_df['cell_id'].isin(valid_rois)]
        peak_results_df = peak_results_df.reset_index(drop=True)

        # --- Debug: Print IDs if none match ---
        if peak_results_df.empty:
            print("CSV IDs:", peak_results_df['cell_id'].unique())
            print("NPY ROI indices:", smoothed_dff_df.index.unique())
            QMessageBox.warning(
                self, "Error",
                "No matching ROIs found between peaks CSV and smoothed data. "
                "Check cell IDs in CSV vs NPY data."
            )
            return
        print("\n>>> ROI LABEL CHECK")
        print("DataFrame ROIs:", smoothed_dff_df.index.tolist())
        print("CSV ROI cell_ids:", peak_results_df['cell_id'].unique().tolist())
        
        common = set(smoothed_dff_df.index).intersection(peak_results_df['cell_id'])
        print("Common ROIs:", sorted(list(common)))
        print("Missing in CSV:", sorted(set(smoothed_dff_df.index) - common))
        print("Missing in Data:", sorted(set(peak_results_df['cell_id']) - common))

        # --- Create the plotter ---
        fig = setup_plot_and_interaction(
            self, smoothed_dff_df, peak_results_df, fs,
            display_window, y_ax_range,
            export_dir, file_prefix, DataLoader(smoothed_dff_df, fs)
        )
        # Initial shading on ax[0]
        self.highlight_interpolated_regions(fig.axes[0], fs)
        # Ensure shading persists during peak navigation
        self.attach_artifact_redraw(fig, fs)

        # Highlight interpolated regions ONLY on ax[0]
        if fig.axes:
            self.highlight_interpolated_regions(fig.axes[0], fs)
        
            
        try:
            fig.canvas.manager.window.activateWindow()
            fig.canvas.manager.window.raise_()
        except Exception as e:
            print(f"Could not raise figure window: {e}")
    
        self.figures.append(fig)
        plt.show()
        

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = CalciumImagingApp()
    window.resize(400, 600)
    window.show()
    sys.exit(app.exec_())