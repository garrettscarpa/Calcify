from matplotlib.widgets import Button
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
        for step in range(1, n + 1):
            candidate_idx = (self.plotter.roi_idx + step) % n
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
        for step in range(1, n + 1):
            candidate_idx = (self.plotter.roi_idx - step) % n
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
    
        # --- Save filtered peaks ---
        filtered_peaks_filename = os.path.join(
            export_dir, f"{file_prefix}_filtered_peaks.csv"
        )
        export_peaks_df.to_csv(filtered_peaks_filename, index=False)
        print(f"Exported filtered peaks to {filtered_peaks_filename}")
    
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
                    times = peak_times[(left_bases >= start) & (left_bases < end)]                    

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
    
                    timelocked_stats.append({
                        'ROI': roi_name,
                        'time_period': label,
                        'peak_count': peak_count,
                        'duration_min': dur_min,
                        'freq_PeaksPerMin': freq,
                        'isi_seconds': isi
                    })
    
        timelocked_stats_df = pd.DataFrame(timelocked_stats)
        timelocked_stats_filename = os.path.join(
            export_dir, f"{file_prefix}_Individual_ROI_Statistics_timelocked.csv"
        )
        timelocked_stats_df.to_csv(timelocked_stats_filename, index=False)
        print(f"Exported timelocked ROI statistics to {timelocked_stats_filename}")
    
    
           
    def setup_buttons(self):
        button_config = [
            ('Add Peak', self.toggle_add_peak_mode, 0.95), 
            ('Reject', self.reject_peak, 0.9),
            ('Undo Reject', self.undo_rejection, 0.85),
            ('Next', self.advance_to_next_peak, 0.8),
            ('Last', self.go_to_last_peak, 0.75),
            ('Export', self.export_csv, 0.7)
        ]
    
        for label, callback, y in button_config:
            ax = self.fig.add_axes([0.8, y, 0.09, 0.04])
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
