from matplotlib.widgets import Button, RangeSlider
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
from history import HistoryManager
import pandas as pd
from scipy.signal import peak_prominences, peak_widths
import re
from PyQt5.QtWidgets import QMessageBox


class InteractionHandler:
    def __init__(self, data_loader, plotter, peak_results_df, smoothed_dff_df, fs,
         display_window, y_ax_range, ca_marker, directory, 
         fig=None, ax1=None, ax2=None, 
         pre_range_input=None, post_range_input=None,
         file_prefix="", truncate_sec=None, polyorder=None, window_length_sec=None, interpolated_regions=None):

        self.plotter = plotter                       
        self.fig, self.ax1, self.ax2 = self.plotter.fig, self.plotter.ax1, self.plotter.ax2
        self.peak_results_df = peak_results_df          
        self.trace = smoothed_dff_df
        self.fs = fs
        self.display_window = display_window
        self.y_ax_range = y_ax_range
        self.ca_marker = ca_marker
        self.directory = directory
        self.add_peak_mode = False
        self.rejected_peaks = []
        self.change_history = []
        self.roi_ids = list(smoothed_dff_df.index)
        self.roi_idx = 0
        self.data_loader = data_loader
        self.file_prefix = file_prefix
        self.truncate_sec = truncate_sec
        self.polyorder = polyorder
        self.window_length_sec = window_length_sec
        self.pre_range_input = pre_range_input
        self.post_range_input = post_range_input
        self.interpolated_regions = interpolated_regions or []
        self.plotter = plotter
        self.fig, self.ax1, self.ax2 = self.plotter.fig, self.plotter.ax1, self.plotter.ax2
        self.fig, self.ax1, self.ax2 = self.plotter.fig, self.plotter.ax1, self.plotter.ax2

        self.refresh_roi_data()
        self.history_manager = HistoryManager()
        self.buttons = []

        # Setup event listeners
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
    @property
    def peak_results_df(self):
        return self.plotter.peak_results_df

    @peak_results_df.setter
    def peak_results_df(self, value):
        self.plotter.peak_results_df = value

    def _get_peaks_for_current_roi(self):
        peaks = self.peak_results_df[self.peak_results_df['cell_id'] == self.current_roi]
        return peaks

    def toggle_add_peak_mode(self, event=None):
        self.add_peak_mode = not self.add_peak_mode
        if self.add_peak_mode:
            print("Add Peak Mode ENABLED")
            if hasattr(self, 'add_peak_button') and self.add_peak_button:
                self.add_peak_button.label.set_text('Adding...')
        else:
            print("Add Peak Mode DISABLED")
            if hasattr(self, 'add_peak_button') and self.add_peak_button:
                self.add_peak_button.label.set_text('Add Peak')
    def show_error_popup(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Invalid Time Range")
        msg_box.setText(message)
        msg_box.exec_()

    def refresh_roi_data(self):
        self.current_roi = self.roi_ids[self.roi_idx]
        self.peaks_in_roi = self._get_peaks_for_current_roi()

    def setup_interactive_plot(self):
        return self.plotter.setup_interactive_plot()
    
    def on_click(self, event):
        if not self.add_peak_mode or event.inaxes != self.ax1 or event.xdata is None or event.ydata is None:
            return
        self.handle_add_peak_click(event.xdata, event.ydata)
        
    def handle_add_peak_click(self, xdata, ydata):
        print(f"Clicked at ({xdata:.2f} s, {ydata:.2f}) to add peak")
        roi = self.plotter.current_roi  # Use the plotter's current ROI
    
        trace = self.data_loader.get_trace(roi)
        times = self.data_loader.get_times()
    
        peak_idx = np.argmin(np.abs(times - xdata))
        peak_time = times[peak_idx]
        peak_value = trace[peak_idx]
    
        # Walk left to find local minimum
        left_base_idx = peak_idx
        for i in range(peak_idx - 1, 0, -1):
            if trace[i] < trace[i - 1] and trace[i] < trace[i + 1]:
                left_base_idx = i
                break
            if trace[i] < trace[left_base_idx]:
                left_base_idx = i
    
        # Walk right to find local minimum
        right_base_idx = peak_idx
        for i in range(peak_idx + 1, len(trace) - 1):
            if trace[i] < trace[i - 1] and trace[i] < trace[i + 1]:
                right_base_idx = i
                break
            if trace[i] < trace[right_base_idx]:
                right_base_idx = i
    
        left_base_time  = times[left_base_idx]
        right_base_time = times[right_base_idx]
        base_value      = min(trace[left_base_idx], trace[right_base_idx])
    
        try:
            prominences = peak_prominences(trace, [peak_idx])[0][0]
        except Exception:
            prominences = peak_value - base_value
    
        half_rise_time  = peak_time - left_base_time
        half_decay_time = right_base_time - peak_time
        time_to_peak    = half_rise_time
    
        new_peak = pd.DataFrame([{
            'cell_id':         roi,
            'peak_time':       peak_time,
            'left_bases':      left_base_time,
            'right_bases':     right_base_time,
            'prominences':     prominences,
            'base_value':      base_value,
            'peak_value':      peak_value,
            'auc':             None,
            'time_to_peak':    time_to_peak,
            'half_rise_time':  half_rise_time,
            'half_decay_time': half_decay_time,
        }])
    
        new_peak = new_peak.reindex(columns=self.peak_results_df.columns)
        self.peak_results_df = pd.concat([self.peak_results_df, new_peak], ignore_index=True)
        self.peak_results_df = self.peak_results_df.sort_values(['cell_id', 'peak_time']).reset_index(drop=True)
        self.plotter.peak_results_df = self.peak_results_df
    
        peaks_in_roi = self.peak_results_df[self.peak_results_df['cell_id'] == roi]
        matching = peaks_in_roi[np.isclose(peaks_in_roi['peak_time'], peak_time, atol=1e-6)]
        if not matching.empty:
            self.plotter.peak_idx = peaks_in_roi.index.get_loc(matching.index[0])
    
        self.update_plot()
        
    def update_plot(self):
        self.refresh_roi_data()
        self.plotter.roi_idx = self.roi_idx
        self.plotter.update_plot()

    def reject_peak(self, event=None):
        peak_idx = self.plotter.peak_idx
        current_roi = self.plotter.current_roi
        peaks_in_roi = self.plotter._get_peaks_for_current_roi()
    
        if not peaks_in_roi.empty and peak_idx < len(peaks_in_roi):
            peak_row = peaks_in_roi.iloc[peak_idx]
            print(f"Rejecting peak from ROI: {current_roi}, Peak: {peak_row[['cell_id', 'peak_time']].to_dict()}")
    
            self.history_manager.record(
                action="reject",
                index=peak_row.name,
                original_data=peak_row.copy(),
                additional_info=None
            )
    
            self.peak_results_df = self.peak_results_df[
                ~((self.peak_results_df['peak_time'] == peak_row['peak_time']) &
                  (self.peak_results_df['cell_id'] == peak_row['cell_id']))
            ]
    
            self.plotter.peak_results_df = self.peak_results_df
            self.plotter.peaks_in_roi = self.plotter._get_peaks_for_current_roi()
    
            if self.plotter.peak_idx >= len(self.plotter.peaks_in_roi):
                self.plotter.peak_idx = max(0, len(self.plotter.peaks_in_roi) - 1)
    
            self.plotter.update_plot()
        else:
            print("No peaks to reject or index out of bounds.")
        
    def undo_rejection(self, event=None):
        if not self.history_manager.undo_stack:
            print("No undo actions available.")
            return
    
        last_change = self.history_manager.undo_stack.pop()
        if last_change.get('action') != 'reject':
            print("Last action was not a rejection.")
            return
    
        peak_row = last_change['original_data']
        restored_roi = peak_row['cell_id']
        restored_peak_time = peak_row['peak_time']
    
        # Restore the peak and sort
        self.peak_results_df = pd.concat([
            self.peak_results_df,
            peak_row.to_frame().T
        ], ignore_index=True)
        self.peak_results_df = self.peak_results_df.sort_values(
            ['cell_id', 'peak_time']
        ).reset_index(drop=True)
        self.plotter.peak_results_df = self.peak_results_df
    
        # Sync roi_ids on both handler and plotter. Use the full data index
        # so every ROI stays navigable (matches the plotter's all-ROIs list).
        self.roi_ids = list(self.trace.index)
        self.plotter.roi_ids = self.roi_ids
    
        # Navigate to the restored ROI
        if restored_roi in self.roi_ids:
            self.roi_idx = self.roi_ids.index(restored_roi)
        else:
            self.roi_idx = 0
        self.plotter.roi_idx = self.roi_idx
        self.plotter.current_roi = restored_roi
    
        # Rebuild peaks_in_roi for that ROI
        self.plotter.peaks_in_roi = self.plotter._get_peaks_for_current_roi()
    
        # Find the exact position of the restored peak by peak_time
        matching = self.plotter.peaks_in_roi[
            np.isclose(self.plotter.peaks_in_roi['peak_time'], restored_peak_time, atol=0.5 / self.fs)
        ]
        self.plotter.peak_idx = int(matching.index[0]) if not matching.empty else 0
    
        self.plotter.update_plot()

    def advance_to_next_peak(self, event=None):
        # Physical "Next" button: jump to the FIRST peak of the next ROI
        # that actually contains a peak. Operate on the plotter's
        # peaks-only ROI list (the same list the arrow keys use) so the
        # index stays meaningful and we don't skip ROIs.
        roi_ids = self.plotter.roi_ids
        if not roi_ids:
            return
        n = len(roi_ids)
        # Anchor on the ROI actually being displayed, not on a possibly-stale
        # roi_idx. If current_roi and roi_idx have drifted apart, trusting
        # roi_idx makes the scan start from the wrong place and skip an ROI.
        try:
            start_idx = roi_ids.index(self.plotter.current_roi)
        except (ValueError, AttributeError):
            start_idx = self.plotter.roi_idx
        for step in range(1, n + 1):
            candidate_idx = (start_idx + step) % n
            self.plotter.roi_idx = candidate_idx
            self.plotter.current_roi = roi_ids[candidate_idx]
            self.plotter.peaks_in_roi = self.plotter._get_peaks_for_current_roi()
            if not self.plotter.peaks_in_roi.empty:
                self.plotter.peak_idx = 0
                break
        self.roi_idx = self.plotter.roi_idx
        self.plotter.update_plot()

    def go_to_last_peak(self, event=None):
        # Physical "Last" button: jump to the FIRST peak of the previous ROI
        # that actually contains a peak.
        roi_ids = self.plotter.roi_ids
        if not roi_ids:
            return
        n = len(roi_ids)
        # Anchor on the displayed ROI (see advance_to_next_peak).
        try:
            start_idx = roi_ids.index(self.plotter.current_roi)
        except (ValueError, AttributeError):
            start_idx = self.plotter.roi_idx
        for step in range(1, n + 1):
            candidate_idx = (start_idx - step) % n
            self.plotter.roi_idx = candidate_idx
            self.plotter.current_roi = roi_ids[candidate_idx]
            self.plotter.peaks_in_roi = self.plotter._get_peaks_for_current_roi()
            if not self.plotter.peaks_in_roi.empty:
                self.plotter.peak_idx = 0
                break
        self.roi_idx = self.plotter.roi_idx
        self.plotter.update_plot()

    def update_base_value(self, x):
        left_base_idx = int(x * self.fs)
        new_base_value = self.trace[left_base_idx]
        current_base_value = self.trace[left_base_idx]
        self.history_manager.record(
            action="move_base",
            index=left_base_idx,
            original_data=current_base_value,
            additional_info=current_base_value
        )
        self.trace[left_base_idx] = new_base_value
        print(f"Base value updated at index {left_base_idx}: {new_base_value}")
                
    def export_csv(self, event=None):
        export_dir = getattr(self, 'directory', ".")
        file_prefix = getattr(self, 'file_prefix', "exported_data")
    
        # --- Backfill AUC / base metrics for any peak missing them ---
        # Peaks added by clicking get 'auc': None and are never recomputed
        # unless a base marker is dragged. Fill them in before anything
        # downstream (peak table AND timelocked stats) reads the values.
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        _times = np.arange(self.trace.shape[1]) / float(self.fs)
        for _i, _r in self.peak_results_df.iterrows():
            if not pd.isna(_r.get('auc')):
                continue
            if pd.isna(_r.get('left_bases')) or pd.isna(_r.get('right_bases')):
                continue
            _tr = self.trace.loc[_r['cell_id']].values
            _lo = max(0, int(round(_r['left_bases'] * self.fs)))
            _hi = min(len(_tr) - 1, int(round(_r['right_bases'] * self.fs)))
            if _hi <= _lo:
                continue
            _seg, _seg_t = _tr[_lo:_hi + 1], _times[_lo:_hi + 1]
            _bv = min(_tr[_lo], _tr[_hi])
            _pv = _seg.max()
            self.peak_results_df.at[_i, 'auc'] = _trapz(
                np.abs(np.maximum(_seg, _bv) - _bv), _seg_t)
            self.peak_results_df.at[_i, 'base_value']  = _bv
            self.peak_results_df.at[_i, 'peak_value']  = _pv
            self.peak_results_df.at[_i, 'prominences'] = _pv - _bv

        _n = self.peak_results_df['auc'].notna().sum()
        print(f"[export] AUC present for {_n}/{len(self.peak_results_df)} peaks")

        # --- Base peak table ---
        export_peaks_df = self.peak_results_df.copy()
        unique_cell_ids = export_peaks_df['cell_id'].unique()
    
        # --- Assign a per-ROI peak number ---
        peak_number_list = []
        for roi in unique_cell_ids:
            roi_peaks = export_peaks_df[export_peaks_df['cell_id'] == roi] \
                .sort_values('peak_time')
            for i, idx in enumerate(roi_peaks.index):
                peak_number_list.append((idx, i + 1))

        peak_number_dict = dict(peak_number_list)
        export_peaks_df['peak_number'] = export_peaks_df.index.map(peak_number_dict)
    
        # Reorder columns
        cols = list(export_peaks_df.columns)
        cols.remove('peak_number')
        cell_id_idx = cols.index('cell_id')
        cols.insert(cell_id_idx + 1, 'peak_number')
        export_peaks_df = export_peaks_df[cols]
    
        # Add preprocessing params
        export_peaks_df['truncate_sec'] = getattr(self, 'truncate_sec', None)
        export_peaks_df['polyorder'] = getattr(self, 'polyorder', None)
        export_peaks_df['window_length_sec'] = getattr(self, 'window_length_sec', None)
    
        # --- Normalize cell_id formats ---
        def safe_transform(x):
            # Numeric value
            if isinstance(x, (int, float, np.integer, np.floating)):
                return f"ROI_{int(x)}"
            # Numeric string
            if isinstance(x, str) and x.isdigit():
                return f"ROI_{int(x)}"
            # Any other string: leave unchanged
            return str(x)
    
        export_peaks_df['cell_id'] = export_peaks_df['cell_id'].apply(safe_transform)
    
        # --- Numeric sort key from the trailing ROI number (so ROI_2 < ROI_10) ---
        def roi_sort_key(cell_id):
            m = re.search(r'(\d+)$', str(cell_id))
            return int(m.group(1)) if m else np.inf

        # --- Sort filtered peaks by ROI (smallest to largest), then peak_time ---
        export_peaks_df['_roi_order'] = export_peaks_df['cell_id'].apply(roi_sort_key)
        export_peaks_df = export_peaks_df.sort_values(
            ['_roi_order', 'peak_time']
        ).drop(columns='_roi_order').reset_index(drop=True)
    
        # --- Save filtered peaks ---
        filtered_peaks_filename = os.path.join(
            export_dir, f"{file_prefix}_filtered_peaks.csv"
        )
        export_peaks_df.to_csv(filtered_peaks_filename, index=False)
        print(f"Exported filtered peaks to {filtered_peaks_filename}")
    
        # --- Save averaged filtered peaks (per-ROI means, sorted by ROI) ---
        avg_cols = ['prominences', 'auc', 'half_rise_time', 'half_decay_time']
        averaged_peaks_df = (
            export_peaks_df
            .groupby('cell_id', as_index=False)[avg_cols]
            .mean()
        )
        averaged_peaks_df['_roi_order'] = averaged_peaks_df['cell_id'].apply(roi_sort_key)
        averaged_peaks_df = averaged_peaks_df.sort_values(
            '_roi_order'
        ).drop(columns='_roi_order').reset_index(drop=True)
        averaged_peaks_filename = os.path.join(
            export_dir, f"{file_prefix}_filtered_peaks_averaged.csv"
        )
        averaged_peaks_df.to_csv(averaged_peaks_filename, index=False)
        print(f"Exported averaged filtered peaks to {averaged_peaks_filename}")
    
        # ---------------------------------------------------------------------
        # ROI statistics
        # ---------------------------------------------------------------------
    
        # Full ΔF/F data is stored in self.trace (DataFrame)
        all_rois = self.trace.index
        num_samples = self.trace.shape[1]
    
        # Build the time vector
        full_times = np.arange(num_samples) / float(self.fs)
        duration_sec = full_times[-1] if len(full_times) > 1 else 0
        duration_min = duration_sec / 60 if duration_sec > 0 else np.nan
    
        roi_stats = []
        for roi in all_rois:
            roi_name = safe_transform(roi)
            peaks = self.peak_results_df[
                self.peak_results_df['cell_id'] == roi
            ].sort_values('peak_time')
    
            peak_times = peaks['peak_time'].values
            peak_count = len(peak_times)
    
            if peak_count > 1:
                freq = peak_count / duration_min
                isi = np.mean(np.diff(peak_times))
            elif peak_count == 1:
                freq = 1 / duration_min
                isi = float('nan')
            else:
                freq = 0
                isi = float('nan')
    
            roi_stats.append({
                'ROI': roi_name,
                'peak_count': peak_count,
                'duration_min': duration_min,
                'freq_PeaksPerMin': freq,
                'isi_seconds': isi
            })
    
        roi_stats_df = pd.DataFrame(roi_stats)
        roi_stats_filename = os.path.join(
            export_dir, f"{file_prefix}_Individual_ROI_Statistics.csv"
        )
        roi_stats_df.to_csv(roi_stats_filename, index=False)
        print(f"Exported Individual ROI Statistics to {roi_stats_filename}")
    
        # ---------------------------------------------------------------------
        # Timelocked analysis
        # ---------------------------------------------------------------------
    
        def parse_range(text):
            match = re.match(r"^\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*$", text)
            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                if start < end:
                    return start, end
            return None
    
        pre_range = parse_range(self.pre_range_input.text().strip()) \
            if self.pre_range_input.text().strip() else None
        post_range = parse_range(self.post_range_input.text().strip()) \
            if self.post_range_input.text().strip() else None
    
        def range_out_of_bounds(rng, max_dur):
            if not rng:
                return False
            return rng[0] < 0 or rng[1] > max_dur
    
        # Validate time ranges
        if range_out_of_bounds(pre_range, duration_sec):
            self.show_error_popup(
                f"Pre range {pre_range} lies outside total recording duration ({duration_sec:.2f} seconds)."
            )
            return
        if range_out_of_bounds(post_range, duration_sec):
            self.show_error_popup(
                f"Post range {post_range} lies outside total recording duration ({duration_sec:.2f} seconds)."
            )
            return
    
        if not pre_range and not post_range:
            return  # No timelocked analysis requested
    
        timelocked_peaks_df = export_peaks_df.copy()
    
        def label_time_period(t):
            if pre_range and pre_range[0] <= t < pre_range[1]:
                return 'pre'
            if post_range and post_range[0] <= t < post_range[1]:
                return 'post'
            return 'outside'
    
        timelocked_peaks_df['time_period'] = timelocked_peaks_df['left_bases'].apply(label_time_period)
        timelocked_peaks_df = timelocked_peaks_df[timelocked_peaks_df['time_period'] != 'outside']
    
        timelocked_peaks_filename = os.path.join(
            export_dir, f"{file_prefix}_filtered_peaks_timelocked.csv"
        )
        timelocked_peaks_df.to_csv(timelocked_peaks_filename, index=False)
        print(f"Exported timelocked filtered peaks to {timelocked_peaks_filename}")
    
        # --- Overwrite averaged file with per-period averages (sorted by ROI) ---
        averaged_timelocked_df = (
            timelocked_peaks_df
            .groupby(['cell_id', 'time_period'], as_index=False)[avg_cols]
            .mean()
        )
        averaged_timelocked_df['_roi_order'] = averaged_timelocked_df['cell_id'].apply(roi_sort_key)
        averaged_timelocked_df = averaged_timelocked_df.sort_values(
            ['_roi_order', 'time_period']
        ).drop(columns='_roi_order').reset_index(drop=True)
        averaged_timelocked_df.to_csv(averaged_peaks_filename, index=False)
        print(f"Exported per-period averaged filtered peaks to {averaged_peaks_filename}")
    
        # --- Timelocked statistics ---
        timelocked_stats = []
        for roi in all_rois:
            roi_name = safe_transform(roi)
            peaks = self.peak_results_df[
                self.peak_results_df['cell_id'] == roi
            ].sort_values('peak_time')
            peak_times = peaks['peak_time'].values
    
            for label, r in [('pre', pre_range), ('post', post_range)]:
                if r:
                    start, end = r
                    left_bases = peaks['left_bases'].values
                    in_window = (left_bases >= start) & (left_bases < end)
                    times = peak_times[in_window]
                    win_peaks = peaks[in_window]

                    dur_min = (end - start) / 60
                    peak_count = len(times)

                    if peak_count > 1:
                        freq = peak_count / dur_min
                        isi = np.mean(np.diff(times))
                    elif peak_count == 1:
                        freq = 1 / dur_min
                        isi = float('nan')
                    else:
                        freq = 0
                        isi = float('nan')

                    if peak_count > 0:
                        mean_auc = win_peaks['auc'].mean()
                        total_auc = win_peaks['auc'].sum()
                        mean_amp = (win_peaks['peak_value'] - win_peaks['base_value']).mean()
                        max_amp = (win_peaks['peak_value'] - win_peaks['base_value']).max()
                        mean_prom = win_peaks['prominences'].mean()
                    else:
                        mean_auc = total_auc = mean_amp = max_amp = mean_prom = float('nan')

                    timelocked_stats.append({
                        'ROI': roi_name,
                        'time_period': label,
                        'peak_count': peak_count,
                        'duration_min': dur_min,
                        'freq_PeaksPerMin': freq,
                        'isi_seconds': isi,
                        'mean_auc': mean_auc,
                        'total_auc': total_auc,
                        'auc_per_min': total_auc / dur_min if peak_count > 0 else 0.0,
                        'mean_amplitude': mean_amp,
                        'max_amplitude': max_amp,
                        'mean_prominence': mean_prom,
                    })
    
        timelocked_stats_df = pd.DataFrame(timelocked_stats)
        timelocked_stats_filename = os.path.join(
            export_dir, f"{file_prefix}_Individual_ROI_Statistics_timelocked.csv"
        )
        timelocked_stats_df.to_csv(timelocked_stats_filename, index=False)
        print(f"Exported timelocked ROI statistics to {timelocked_stats_filename}")
    
    
           
    def export_peak_rois(self, event=None, show_dialog=True):
        """
        Export a CSV where each ROI that contains at least one peak is a row:
        the first column is the ROI name, and the remaining columns are that
        ROI's raw (full) trace values.

        show_dialog : if True, show the completion popup. Set False for silent
            auto-exports (e.g. triggered on arrow-key navigation) so the user
            isn't constantly dismissing a confirmation window.
        """
        export_dir = getattr(self, 'directory', ".")
        file_prefix = getattr(self, 'file_prefix', "exported_data")

        # Same ROI-name normalization used in export_csv, so names match.
        def safe_transform(x):
            if isinstance(x, (int, float, np.integer, np.floating)):
                return f"ROI_{int(x)}"
            if isinstance(x, str) and x.isdigit():
                return f"ROI_{int(x)}"
            return str(x)

        # ROIs that actually contain a peak, in trace order.
        rois_with_peaks = set(self.peak_results_df['cell_id'].unique())
        ordered_rois = [roi for roi in self.trace.index if roi in rois_with_peaks]

        if not ordered_rois:
            if show_dialog:
                QMessageBox.warning(None, "No Peaks",
                                    "No ROIs contain peaks to export.")
            return

        # Build one row per ROI: [ROI name, raw sample 0, raw sample 1, ...]
        rows = self.trace.loc[ordered_rois].copy()
        rows.insert(0, 'ROI', [safe_transform(r) for r in ordered_rois])

        # Name the sample columns clearly.
        n_samples = self.trace.shape[1]
        rows.columns = ['ROI'] + [f"sample_{i}" for i in range(n_samples)]

        out_filename = os.path.join(
            export_dir, f"{file_prefix}_peak_ROI_traces.csv"
        )
        rows.to_csv(out_filename, index=False)
        print(f"Exported peak ROI traces to {out_filename}")
        if show_dialog:
            QMessageBox.information(
                None, "Export Complete",
                f"Exported {len(ordered_rois)} ROI trace(s) to:\n{out_filename}"
            )

    def export_all(self, event=None, show_dialog=True):
        """Combined export: writes the peak tables / ROI statistics (export_csv)
        AND the per-ROI raw trace CSV (export_peak_rois) in one action. They
        write distinct files to the same folder, so running both is safe.
        export_peak_rois shows the single completion dialog at the end.

        show_dialog : pass False for silent auto-exports (arrow-key navigation)
            so no confirmation window pops up."""
        self.export_csv(event)
        self.export_peak_rois(event, show_dialog=show_dialog)

    def auto_export(self):
        """Silent save used for autosave-on-navigation. Runs the same export
        as the Export button but suppresses all popups and never raises, so a
        failed save can't interrupt the user's workflow."""
        try:
            self.export_all(show_dialog=False)
        except Exception as e:
            # Autosave must never crash navigation; just log it.
            print(f"[auto_export] skipped due to error: {e}")

    # ------------------------------------------------------------------
    # Representative-peak summary
    # ------------------------------------------------------------------
    def _parse_prepost_ranges(self):
        """Return (pre_range, post_range) as (start, end) tuples in seconds, or
        None for each if the corresponding input is blank/invalid. Mirrors the
        parser used in export_csv so the summary matches the timelocked export."""
        def parse_range(text):
            m = re.match(r"^\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*$", text)
            if m:
                start, end = float(m.group(1)), float(m.group(2))
                if start < end:
                    return start, end
            return None

        pre = post = None
        if self.pre_range_input is not None:
            txt = self.pre_range_input.text().strip()
            pre = parse_range(txt) if txt else None
        if self.post_range_input is not None:
            txt = self.post_range_input.text().strip()
            post = parse_range(txt) if txt else None
        return pre, post

    def _extract_aligned_peaks(self, peaks_df, pre_pad_s, post_pad_s, expand=1.0):
        """Extract each peak from its ROI trace onto a common time grid centred
        on the peak (t=0 at the peak). Returns (t_grid, matrix) where matrix is
        (n_peaks x n_timepoints).

        A peak is kept when its DEFAULT window (pre_pad_s before to post_pad_s
        after the peak) fits inside the recording. The extra width added by
        `expand` (there so the zoom-out slider has data to reveal) is allowed
        to run off the recording: those samples are left as NaN and the
        summary statistics are NaN-aware. (Previously a peak was dropped if the
        *expanded* window didn't fit, so with slow transients - where the
        median peak-to-base distance is tens of seconds - the 3x window was
        longer than the whole recording and every peak was discarded.)

        The expanded grid is also capped so it is never longer than the
        recording itself.

        Amplitude is baseline-subtracted per peak (base = min of the two bases),
        so the average represents dF/F above baseline and shading reflects
        peak-to-peak variability rather than differing DC offsets."""
        n_rec = self.trace.shape[1]
        tight_pre = int(round(pre_pad_s * self.fs))
        tight_post = int(round(post_pad_s * self.fs))

        # Cap the expansion so the whole grid fits within the recording length.
        expand_eff = max(1.0, min(float(expand),
                                  n_rec / float(tight_pre + tight_post + 1)))
        pre_n = int(round(pre_pad_s * expand_eff * self.fs))
        post_n = int(round(post_pad_s * expand_eff * self.fs))
        t_grid = np.arange(-pre_n, post_n + 1) / self.fs
        L = t_grid.size

        rows = []
        for _, r in peaks_df.iterrows():
            cell = r['cell_id']
            if cell not in self.trace.index:
                continue
            tr = self.trace.loc[cell].values.astype(float)
            peak_idx = int(round(r['peak_time'] * self.fs))

            # Require the default (tight) window to fit fully.
            if peak_idx - tight_pre < 0 or peak_idx + tight_post >= len(tr):
                continue

            # Copy the overlapping part of the expanded window; NaN elsewhere.
            lo, hi = peak_idx - pre_n, peak_idx + post_n
            src_lo, src_hi = max(lo, 0), min(hi, len(tr) - 1)
            seg = np.full(L, np.nan)
            seg[src_lo - lo: src_lo - lo + (src_hi - src_lo + 1)] = tr[src_lo:src_hi + 1]

            # Baseline = min of the two base values (same convention as detection)
            lb, rb = r.get('left_bases'), r.get('right_bases')
            if pd.notna(lb) and pd.notna(rb):
                lb_i = int(np.clip(round(lb * self.fs), 0, len(tr) - 1))
                rb_i = int(np.clip(round(rb * self.fs), 0, len(tr) - 1))
                base = min(tr[lb_i], tr[rb_i])
            else:
                base = np.nanmin(seg)
            rows.append(seg - base)

        if not rows:
            return t_grid, np.empty((0, t_grid.size))
        return t_grid, np.vstack(rows)

    def _window_padding(self, peaks_df):
        """Choose a symmetric extraction window from the peak set.

        Starts from the median gap from peak to each base (so the representative
        peak captures a typical rise and decay), then shrinks that window - but
        never below 50% of the median on either side - to the size that lets the
        MOST peaks fit fully inside the recording.

        Why: find_peaks places bases at the lowest points before a higher peak or
        the trace edge, so for slow/isolated events the bases can sit at the very
        start and end of the recording. The median window can then be nearly as
        long as the recording itself, and only peaks in a narrow central band fit
        (in one real dataset: 1 of 6 peaks). Fitting the window to the recording
        keeps far more peaks in the average."""
        rise = (peaks_df['peak_time'] - peaks_df['left_bases']).dropna()
        decay = (peaks_df['right_bases'] - peaks_df['peak_time']).dropna()
        # Fall back to a small fixed window if bases are missing.
        pre_pad = float(np.nanmedian(rise)) if len(rise) else 2.0
        post_pad = float(np.nanmedian(decay)) if len(decay) else 4.0
        # Guard against degenerate/zero windows.
        pre_pad = max(pre_pad, 1.0 / self.fs * 5)
        post_pad = max(post_pad, 1.0 / self.fs * 5)

        # ---- shrink to fit the recording (see docstring) ----
        n_rec = self.trace.shape[1]
        pt = pd.to_numeric(peaks_df['peak_time'], errors='coerce').dropna()
        idx = np.round(pt.values * self.fs).astype(int)
        idx = idx[(idx >= 0) & (idx < n_rec)]
        if idx.size == 0:
            return pre_pad, post_pad

        pre_med_n = max(1, int(round(pre_pad * self.fs)))
        post_med_n = max(1, int(round(post_pad * self.fs)))
        pre_floor = min(pre_med_n, max(5, int(np.ceil(0.5 * pre_med_n))))
        post_floor = min(post_med_n, max(5, int(np.ceil(0.5 * post_med_n))))

        room_pre = idx                      # samples available before each peak
        room_post = n_rec - 1 - idx         # samples available after each peak

        pre_cands = np.unique(np.clip(np.append(room_pre, pre_med_n),
                                      pre_floor, pre_med_n))
        post_cands = np.unique(np.clip(np.append(room_post, post_med_n),
                                       post_floor, post_med_n))

        best = (-1, -1, pre_med_n, post_med_n)     # (kept, total window, pre, post)
        for a in pre_cands:
            sub = np.sort(room_post[room_pre >= a])
            if sub.size == 0:
                continue
            # peaks kept for each candidate post window b: those with room_post >= b
            kept = sub.size - np.searchsorted(sub, post_cands, side='left')
            for b, k in zip(post_cands, kept):
                cand = (int(k), int(a + b), int(a), int(b))
                if cand[:2] > best[:2]:
                    best = cand
        if best[0] > 0:
            pre_pad, post_pad = best[2] / self.fs, best[3] / self.fs
        return pre_pad, post_pad

    def _draw_representative_axis(self, ax, t, mat, color, title, shade='sem'):
        """Plot mean ± error band for one aligned peak matrix on a publication
        styled axis. Every panel uses the same palette: a black mean trace with
        gray error shading. `color` is accepted for backward compatibility but
        no longer varies the palette. shade: 'sem' (default) or 'std'.

        Stats are nan-aware because peaks near the recording edge may be padded
        with NaN when the extraction window is widened for zoom-out."""
        TRACE_COLOR = 'black'
        SHADE_COLOR = '0.6'   # medium gray

        # Per-timepoint count of contributing peaks (varies near the edges).
        counts = np.sum(~np.isnan(mat), axis=0)
        n = int(counts.max()) if counts.size else 0
        mean = np.nanmean(mat, axis=0)
        sd = np.nanstd(mat, axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)
        if shade == 'std':
            err = sd
            err_label = 'SD'
        else:
            with np.errstate(invalid='ignore', divide='ignore'):
                err = sd / np.sqrt(np.maximum(counts, 1))
            err_label = 'SEM'

        ax.axvline(0, color='0.8', lw=0.8, ls='--', zorder=1)
        ax.fill_between(t, mean - err, mean + err, color=SHADE_COLOR, alpha=0.4,
                        linewidth=0, zorder=2, label=f'± {err_label}')
        ax.plot(t, mean, color=TRACE_COLOR, lw=2.0, zorder=3)

        ax.set_title(f'{title}  (n = {n} peaks)', fontsize=11)
        ax.set_xlabel('Time from peak (s)', fontsize=11)
        ax.set_ylabel('ΔF/F (baseline-subtracted)', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.legend(frameon=False, fontsize=9, loc='upper right')

    def summarize_peaks(self, event=None, shade='sem'):
        """Build a publication-quality representative-peak figure.

        Mode 1 (no pre/post ranges): a single panel averaging every peak from
        every ROI, with error shading.
        Mode 2 (pre and/or post ranges defined): one panel per defined period,
        each averaging the peaks whose left base falls in that window, using the
        same alignment and shading rules as Mode 1.

        Peaks are aligned at their peak time and baseline-subtracted before
        averaging. The figure is shown and saved next to the data as
        <prefix>_representative_peak(s).png/.svg.
        """
        df = self.peak_results_df
        if df is None or df.empty:
            QMessageBox.warning(None, "No Peaks",
                                "There are no peaks to summarize. Process the "
                                "data and detect peaks first.")
            return

        pre_range, post_range = self._parse_prepost_ranges()

        export_dir = getattr(self, 'directory', ".")
        file_prefix = getattr(self, 'file_prefix', "exported_data")

        # ---- Mode 2: pre/post defined -> one panel per defined period ----
        if pre_range or post_range:
            periods = []
            if pre_range:
                periods.append(('Pre', pre_range, '#4C8BB5'))
            if post_range:
                periods.append(('Post', post_range, '#E06B4A'))

            def in_range(rng):
                lb = df['left_bases']
                return df[(lb >= rng[0]) & (lb < rng[1])]

            # Shared window across periods so panels are directly comparable.
            all_sel = pd.concat([in_range(r) for _, r, _ in periods]) \
                if periods else df
            if all_sel.empty:
                QMessageBox.warning(None, "No Peaks in Range",
                                    "No peaks fall within the defined pre/post "
                                    "time range(s).")
                return
            pre_pad, post_pad = self._window_padding(all_sel)
            # Extract a wider grid than the default view so the slider can zoom
            # OUT (reveal more) as well as in. The initial x-limits show the
            # tight window; the slider spans the full extracted extent.
            EXPAND = 3.0

            panels = []
            for name, rng, color in periods:
                sel = in_range(rng)
                t, mat = self._extract_aligned_peaks(sel, pre_pad, post_pad,
                                                     expand=EXPAND)
                panels.append((name, rng, color, t, mat))

            usable = [p for p in panels if p[4].shape[0] > 0]
            if not usable:
                QMessageBox.warning(None, "No Peaks in Range",
                                    "Peaks were found in the time range(s) but "
                                    "none fit fully within the averaging window "
                                    f"(-{pre_pad:.1f} s to +{post_pad:.1f} s around "
                                    f"each peak; recording is "
                                    f"{self.trace.shape[1] / self.fs:.1f} s). The "
                                    "window comes from the median distance to "
                                    "each peak's bases, which is long for slow "
                                    "transients.")
                return

            fig, axes = plt.subplots(1, len(usable),
                                     figsize=(5.0 * len(usable), 4.5),
                                     squeeze=False)
            axes = axes[0]

            # Share y-limits so pre vs post amplitude is visually comparable.
            # Restrict the y-scan to the default (tight) view so the wide zoom-out
            # tails don't blow up the y-range. nan-aware for padded edge peaks.
            ymins, ymaxs = [], []
            for name, rng, color, t, mat in usable:
                view = (t >= -pre_pad) & (t <= post_pad)
                mean = np.nanmean(mat, axis=0)
                cnt = np.sum(~np.isnan(mat), axis=0)
                sd = np.nanstd(mat, axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)
                with np.errstate(invalid='ignore', divide='ignore'):
                    err = sd / np.sqrt(np.maximum(cnt, 1)) if shade == 'sem' else sd
                lo_band, hi_band = (mean - err)[view], (mean + err)[view]
                if lo_band.size:
                    ymins.append(np.nanmin(lo_band))
                    ymaxs.append(np.nanmax(hi_band))
            ylo, yhi = (min(ymins), max(ymaxs)) if ymins else (0.0, 1.0)
            pad = 0.08 * (yhi - ylo if yhi > ylo else 1.0)

            summary_axes = []
            t_all = usable[0][3]
            for ax, (name, rng, color, t, mat) in zip(axes, usable):
                self._draw_representative_axis(
                    ax, t, mat, color,
                    f'{name} ({rng[0]:g}–{rng[1]:g} s)', shade=shade)
                ax.set_ylim(ylo - pad, yhi + pad)
                ax.set_xlim(-pre_pad, post_pad)   # default (tight) view
                summary_axes.append(ax)

            fig.suptitle(f'{file_prefix} — representative peak by period',
                         fontsize=12)
            out_base = os.path.join(
                export_dir, f"{file_prefix}_representative_peaks")
            default_view = (-pre_pad, post_pad)

        # ---- Mode 1: no ranges -> single averaged peak across all ROIs ----
        else:
            pre_pad, post_pad = self._window_padding(df)
            EXPAND = 3.0
            t, mat = self._extract_aligned_peaks(df, pre_pad, post_pad,
                                                 expand=EXPAND)
            if mat.shape[0] == 0:
                QMessageBox.warning(None, "No Peaks",
                                    "No peaks fit fully within the averaging "
                                    f"window (-{pre_pad:.1f} s to +{post_pad:.1f} s "
                                    f"around each peak; recording is "
                                    f"{self.trace.shape[1] / self.fs:.1f} s). The "
                                    "window comes from the median distance to "
                                    "each peak's bases, which is long for slow "
                                    "transients.")
                return
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            self._draw_representative_axis(
                ax, t, mat, '#3A3A98',
                'All ROIs', shade=shade)

            # Scale y to the default (tight) view, not the zoom-out tails.
            view = (t >= -pre_pad) & (t <= post_pad)
            mean = np.nanmean(mat, axis=0)
            cnt = np.sum(~np.isnan(mat), axis=0)
            sd = np.nanstd(mat, axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)
            with np.errstate(invalid='ignore', divide='ignore'):
                err = sd / np.sqrt(np.maximum(cnt, 1)) if shade == 'sem' else sd
            if view.any():
                ylo = np.nanmin((mean - err)[view])
                yhi = np.nanmax((mean + err)[view])
                pad = 0.08 * (yhi - ylo if yhi > ylo else 1.0)
                ax.set_ylim(ylo - pad, yhi + pad)
            ax.set_xlim(-pre_pad, post_pad)   # default (tight) view

            fig.suptitle(f'{file_prefix} — representative peak (all ROIs)',
                         fontsize=12)
            out_base = os.path.join(
                export_dir, f"{file_prefix}_representative_peak")
            summary_axes = [ax]
            t_all = t
            default_view = (-pre_pad, post_pad)

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Save publication-quality raster + vector copies BEFORE adding the
        # slider, so the exported figures stay clean (no widget in them).
        png_path = out_base + ".png"
        svg_path = out_base + ".svg"
        try:
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            fig.savefig(svg_path, bbox_inches='tight')
            print(f"[summarize_peaks] saved {png_path}")
            print(f"[summarize_peaks] saved {svg_path}")
        except Exception as e:
            print(f"[summarize_peaks] could not save figure: {e}")

        # ---- Interactive x-axis range slider ----
        # The slider spans the full EXTRACTED grid (wider than the default view),
        # so dragging the handles zooms both IN (narrower than default) and OUT
        # (revealing the wider extracted tails). It initialises at the default
        # tight view. The reference is stashed on the figure so the widget isn't
        # garbage-collected.
        self._add_xrange_slider(fig, summary_axes, t_all, default_view)

        plt.show()

    def _add_xrange_slider(self, fig, axes, t, default_view=None):
        """Attach a RangeSlider under the summary panels that controls the
        shared x-axis (time-from-peak) limits of all panels.

        t : the full extracted time grid; the slider's travel spans its extent
            so the user can zoom out past the default view.
        default_view : (lo, hi) initial handle positions. Defaults to the full
            extent if not given."""
        tmin, tmax = float(np.min(t)), float(np.max(t))
        if not np.isfinite(tmin) or not np.isfinite(tmax) or tmin == tmax:
            return  # nothing sensible to scale

        if default_view is None:
            init_lo, init_hi = tmin, tmax
        else:
            init_lo, init_hi = default_view
            # Clamp the initial view inside the slider's travel.
            init_lo = max(tmin, min(init_lo, tmax))
            init_hi = max(tmin, min(init_hi, tmax))
            if init_hi <= init_lo:
                init_lo, init_hi = tmin, tmax

        # Make room at the bottom for the slider without squashing the axes too
        # much; leave the saved figure untouched (already written above).
        fig.subplots_adjust(bottom=0.24)
        slider_ax = fig.add_axes([0.25, 0.08, 0.5, 0.04])
        slider = RangeSlider(
            slider_ax, "x-range (s)",
            valmin=tmin, valmax=tmax,
            valinit=(init_lo, init_hi),
            valfmt="%.2f",
        )

        def _update(val):
            lo, hi = val
            if hi <= lo:
                return
            for ax in axes:
                ax.set_xlim(lo, hi)
            fig.canvas.draw_idle()

        slider.on_changed(_update)
        # Retain references so the widget stays alive for the figure's lifetime.
        if not hasattr(fig, "_kept_widgets"):
            fig._kept_widgets = []
        fig._kept_widgets.append(slider)
        self._xrange_slider = slider

    def setup_buttons(self):
        button_config = [
            ('Add Peak', self.toggle_add_peak_mode, 0.95), 
            ('Reject Peak', self.reject_peak, 0.9),
            ('Undo Reject', self.undo_rejection, 0.85),
            ('Next ROI', self.advance_to_next_peak, 0.8),
            ('Last ROI', self.go_to_last_peak, 0.75),
            ('Export', self.export_all, 0.7),
            ('Summarize Peaks', self.summarize_peaks, 0.65),
        ]
    
        for label, callback, y in button_config:
            # All buttons share one size and sit in the right margin
            # (the plots end at right=0.79), right-aligned with a small gap.
            width = 0.15
            x = 0.83
            ax = self.fig.add_axes([x, y, width, 0.04])
            button = Button(ax, label)
            button.on_clicked(callback)
            self.buttons.append(button)  # Retain reference to prevent garbage collection
    
        # Keep a reference to the Add Peak button so its label can be toggled
        self.add_peak_button = self.buttons[0]

        
    def on_key(self, event):
        if event.key == 'right':
            if self.plotter.peak_idx < len(self.plotter.peaks_in_roi) - 1:
                self.plotter.peak_idx += 1
            else:
                self.plotter.roi_idx = (self.plotter.roi_idx + 1) % len(self.plotter.roi_ids)
                self.plotter.current_roi = self.plotter.roi_ids[self.plotter.roi_idx]
                self.plotter.peaks_in_roi = self.plotter._get_peaks_for_current_roi()
                self.plotter.peak_idx = 0
    
        elif event.key == 'left':
            if self.plotter.peak_idx > 0:
                self.plotter.peak_idx -= 1
            else:
                self.plotter.roi_idx = (self.plotter.roi_idx - 1) % len(self.plotter.roi_ids)
                self.plotter.current_roi = self.plotter.roi_ids[self.plotter.roi_idx]
                self.plotter.peaks_in_roi = self.plotter._get_peaks_for_current_roi()
                self.plotter.peak_idx = max(0, len(self.plotter.peaks_in_roi) - 1)
    
        # Sync handler's roi_idx to match plotter
        self.roi_idx = self.plotter.roi_idx
        self.update_plot()

        # Autosave progress on every arrow-key navigation so a freeze/crash
        # doesn't lose the user's edits. Silent (no popups) and never raises.
        if event.key in ('left', 'right'):
            self.auto_export()
