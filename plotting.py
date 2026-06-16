import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from draggable_point import DraggablePoint
from peak_interaction_handler import InteractionHandler
from custom_markers import calcium_trace_symbol

# np.trapz -> np.trapezoid in NumPy 2.0; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


class InteractivePlotter:
    def __init__(self, smoothed_dff_df, peak_results_df, fs, display_window, y_ax_range, ca_marker,
                 fig, ax1, ax2, role = None):
        self.smoothed_dff_df = smoothed_dff_df
        self.peak_results_df = peak_results_df
        self.fs = fs
        self.display_window = display_window
        self.y_ax_range = y_ax_range
        self.ca_marker = ca_marker
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        # Show ALL ROIs from the data, not only those that have detected
        # peaks. ROIs with no peaks still get a trace in ax1 and a
        # "No Peaks Detected" message in ax2 (handled in update_plot).
        self.roi_ids = list(smoothed_dff_df.index)
        self.roi_idx = 0
        self.current_roi = self.roi_ids[self.roi_idx]
        self.peaks_in_roi = self._get_peaks_for_current_roi()
        self.peak_idx = 0
        self.change_history = []
        self.role = role
        

    def setup_interactive_plot(self):
        time = np.arange(self.smoothed_dff_df.shape[1]) / self.fs
        trace = self.smoothed_dff_df.loc[self.current_roi]
        self.ax1.clear()
        self.ax1.plot(time, trace)
        self.ax1.set_title(f"Full Trace: {self.current_roi}")
        #self.ax1.set_ylim(self.y_ax_range)

        if not self.peaks_in_roi.empty:
            peak_row = self.peaks_in_roi.iloc[self.peak_idx]
            peak_time = peak_row['peak_time']

            half_window = self.display_window / 2
            start_time = peak_time - half_window
            end_time = peak_time + half_window

            left_base_time = peak_row['left_bases']
            right_base_time = peak_row['right_bases']

            start_time = min(start_time, left_base_time)
            end_time = max(end_time, right_base_time)

            start_idx = max(0, int(start_time * self.fs))
            end_idx = min(len(trace), int(end_time * self.fs))

            self.ax2.clear()
            self.ax2.plot(time[start_idx:end_idx], trace[start_idx:end_idx])
            self.ax2.set_title(f"Zoomed Peak at {peak_time:.2f}s")

            zoom_vals = trace[start_idx:end_idx]

            ymin, ymax = np.min(zoom_vals), np.max(zoom_vals)
            margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.1
            self.ax2.set_ylim(ymin - margin, ymax + margin)

            if start_time <= left_base_time <= end_time:
                self.ax2.plot(left_base_time, trace.iloc[int(left_base_time * self.fs)], 'ro')   # Left base marker (no legend entry)
            if start_time <= right_base_time <= end_time:
                self.ax2.plot(right_base_time, trace.iloc[int(right_base_time * self.fs)], 'go')  # Right base marker (no legend entry)
        else:
            # First ROI has no detected peaks — show an empty zoom panel.
            self.ax2.clear()
            self.ax2.text(0.5, 0.5, 'No Peaks Detected', ha='center', va='center', fontsize=14)
            self.ax2.axis('off')

        return self.fig, self.ax1, self.ax2, self.roi_ids, self.roi_idx, self.current_roi, self.peaks_in_roi, self.peak_idx
        print(f"[DEBUG] Full trace min/max for {self.current_roi}: {trace.min()}/{trace.max()}")

    def update_plot(self):
        # Keep the full ROI list (all ROIs in the data). Do NOT collapse to
        # only ROIs with peaks, or peakless ROIs would vanish from navigation.
        self.roi_ids = list(self.smoothed_dff_df.index)
        if self.roi_idx >= len(self.roi_ids):
            self.roi_idx = len(self.roi_ids) - 1
        if self.roi_idx < 0:
            self.roi_idx = 0
    
        self.current_roi = self.roi_ids[self.roi_idx]
        trace = self.smoothed_dff_df.loc[self.current_roi].values
        time_axis = np.arange(len(trace)) / self.fs
        self.peaks_in_roi = self._get_peaks_for_current_roi()
    
        # Handle peak index bounds
        if not self.peaks_in_roi.empty:
            if self.peak_idx >= len(self.peaks_in_roi):
                self.peak_idx = len(self.peaks_in_roi) - 1
            elif self.peak_idx < 0:
                self.peak_idx = 0
        else:
            self.peak_idx = 0
    
        # --- ax1: full trace ---
        self.ax1.clear()
        self.ax1.plot(time_axis, trace, label=f'ROI: {self.current_roi}')
        roi_count_str = f'ROI {self.roi_idx + 1} of {len(self.roi_ids)}'
        self.ax1.set_title(f'Full Trace for {roi_count_str} [0-based ID: {self.current_roi}]')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('ΔF/F')
        self.ax1.grid(True)
        self.ax1.set_ylim(self.y_ax_range)
    
        # --- ax2: zoomed peak view or empty-state message ---
        self.ax2.clear()
        roi_num = self.roi_idx + 1
    
        if not self.peaks_in_roi.empty:
            peak_row = self.peaks_in_roi.iloc[self.peak_idx]
            self._plot_peak_region(peak_row, trace, time_axis)
            peak_num = self.peak_idx + 1
            total_peaks = len(self.peaks_in_roi)
            self.ax2.set_title(f'ROI {roi_num} : Peak {peak_num} of {total_peaks}')
        else:
            self.ax2.text(0.5, 0.5, 'No Peaks Detected', ha='center', va='center', fontsize=14)
            self.ax2.axis('off')
            self.ax2.set_title(f'ROI {roi_num} : No Peaks')
    
        self.fig.canvas.draw()  
    
    def _get_peaks_for_current_roi(self):
        df = self.peak_results_df[
            self.peak_results_df['cell_id'] == self.current_roi
        ]
    
        if 'rejected' in df.columns:
            df = df[~df['rejected']]
    
        df = df.copy()
        df['master_idx'] = df.index
        return df.sort_values('peak_time').reset_index(drop=True)

    def _plot_peak_region(self, peak_row, trace, time_axis):
        print("\n[DEBUG] Plotting peak region for ROI", self.current_roi, "Peak idx", self.peak_idx)
        print("[DEBUG] BEFORE redraw → left_bases, right_bases:", peak_row['left_bases'], peak_row['right_bases'])
        peak_time = peak_row['peak_time']
        center_idx = int(peak_time * self.fs)
        half_window = int(self.display_window * self.fs / 2)
        start_idx = max(center_idx - half_window, 0)
        end_idx = min(center_idx + half_window, len(trace))

        self.ax2.clear()
        self.ax2.plot(time_axis[start_idx:end_idx], trace[start_idx:end_idx])
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('ΔF/F')
        self.ax2.grid(True)

        peak_value = peak_row['peak_value']
        base_value = peak_row['base_value']
        prominence = peak_row['prominences']
        left_base_time = peak_row['left_bases']
        right_base_time = peak_row['right_bases']
        half_rise_time = peak_row.get('half_rise_time', np.nan)
        half_decay_time = peak_row.get('half_decay_time', np.nan)

        self.ax1.axvline(peak_time, color='r', linestyle='--', label='Zoomed Peak')

        if not np.isnan(half_rise_time):
            half_rise_abs_time = left_base_time + half_rise_time
            y_rise = np.interp(half_rise_abs_time, time_axis, trace)
            self.ax2.plot(half_rise_abs_time, y_rise, marker='+', color='blue', markersize=6)
            self.ax2.plot([np.nan], [np.nan], '+', color='blue', label=f'Half-Rise [{half_rise_time:.3f} s]')

        if not np.isnan(half_decay_time):
            half_decay_abs_time = peak_time + half_decay_time
            y_decay = np.interp(half_decay_abs_time, time_axis, trace)
            self.ax2.plot(half_decay_abs_time, y_decay, marker='+', color='red', markersize=6)
            self.ax2.plot([np.nan], [np.nan], '+', color='red', label=f'Half-Decay [{half_decay_time:.3f} s]')

        lw = 3
        self.ax2.plot([peak_time - lw / 2, peak_time + lw / 2], [peak_value, peak_value], color='orange', linestyle='--', linewidth=2, label=f'Peak Value [{peak_value:.3f}]')
        self.ax2.plot([peak_time - lw / 2, peak_time + lw / 2], [base_value, base_value], color='g', linestyle='--', linewidth=2, label=f'Base Value [{base_value:.3f}]')
        self.ax2.plot([peak_time, peak_time], [base_value, peak_value], color='k', linestyle='-', linewidth=2, label=f'Prominence [{prominence:.3f}]')

        self.ax2.plot([np.nan], [np.nan], 'o', color='blue', label=f'Start [{left_base_time:.3f} s]')
        self.ax2.plot([np.nan], [np.nan], 'o', color='red', label=f'Stop [{right_base_time:.3f} s]')


        # --- Persistent draggable base markers ---
        self.left_base_marker = DraggablePoint(
            self.ax2, left_base_time, np.interp(left_base_time, time_axis, trace),
            'blue', '_nolegend_',
            time_axis, trace, self.fs,
            self.smoothed_dff_df, self.current_roi, self,
            role='left_base'
        )
        
        self.right_base_marker = DraggablePoint(
            self.ax2, right_base_time, np.interp(right_base_time, time_axis, trace),
            'red', '_nolegend_',
            time_axis, trace, self.fs,
            self.smoothed_dff_df, self.current_roi, self,
            role='right_base'
        )
            
        # --- AUC shading ---
    

        left_idx = int(left_base_time * self.fs)
        right_idx = int(right_base_time * self.fs)
        if 0 <= left_idx < right_idx <= len(trace):
            auc_times = time_axis[left_idx:right_idx + 1]
            auc_values = trace[left_idx:right_idx + 1]
            min_value = min(base_value, peak_value)
            self.ax2.fill_between(auc_times, auc_values, min_value, color='purple', alpha=0.6)
            auc_abs = np.abs(auc_values - base_value)
            auc_value = _trapz(auc_abs, dx=1/self.fs)
            self.ax2.plot([], [], color='purple', alpha=0.6, marker=self.ca_marker,
                          markeredgecolor='blue', markersize=20, linestyle='None',
                          label=f'AUC [{auc_value:.3f}]')




        self.ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        self.fig.canvas.draw_idle()


    def next_roi(self):
        self.roi_idx = (self.roi_idx + 1) % len(self.roi_ids)
        self.peak_idx = 0
        self.update_plot()

    def prev_roi(self):
        self.roi_idx = (self.roi_idx - 1) % len(self.roi_ids)
        self.peak_idx = 0
        self.update_plot()

    def next_peak(self):
        if not self.peaks_in_roi.empty:
            self.peak_idx = (self.peak_idx + 1) % len(self.peaks_in_roi)
            self.update_plot()

    def prev_peak(self):
        if not self.peaks_in_roi.empty:
            self.peak_idx = (self.peak_idx - 1) % len(self.peaks_in_roi)
            self.update_plot()

def plot_all_rois_from_files(file_list, fs, truncate_seconds, transpose, ca_marker, display_window, y_ax_range):

    for file_path in file_list:
        # Load data
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            df_raw = pd.read_csv(file_path)
            data = df_raw.iloc[:, 2:]
        elif ext == '.npy':
            npy_data = np.load(file_path)
            data = pd.DataFrame(npy_data)
        else:
            print(f"Unsupported file type for {file_path}, skipping...")
            continue

        if transpose:
            data = data.T

        # Truncate data
        truncate_samples = int(truncate_seconds * fs)
        if truncate_samples > 0 and truncate_samples < data.shape[1]:
            data = data.iloc[:, truncate_samples:]

        # Normalize data: dF/F calculation
        dff_array = data.values
        baseline = np.nanmean(dff_array, axis=1, keepdims=True)
        baseline[baseline == 0] = np.nan
        dff_array = (dff_array - baseline) / baseline
        dff_array = np.nan_to_num(dff_array)

        all_roi_traces = dff_array
        n_timepoints = all_roi_traces.shape[1]
        time_vector = np.arange(n_timepoints) / fs

        # Create interactive plot
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.subplots_adjust(bottom=0.15)
        ax.set_title(f"All ROI Traces: {os.path.basename(file_path)}")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("dF/F")

        lines = []
        annotation = None  # Only one annotation at a time

        for idx in range(all_roi_traces.shape[0]):
            roi_trace = all_roi_traces[idx]
            line, = ax.plot(time_vector, roi_trace, label=f"ROI_{idx}")
            lines.append(line)

        def on_click(event):
            nonlocal annotation

            if event.inaxes != ax:
                return

            click_x = event.xdata
            click_y = event.ydata

            min_dist = float('inf')
            closest_line = None
            closest_x = None
            closest_y = None

            for line in lines:
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                if len(xdata) != len(ydata):
                    continue

                idx = np.argmin(np.abs(xdata - click_x))
                x_val = xdata[idx]
                y_val = ydata[idx]

                dist = np.hypot(x_val - click_x, y_val - click_y)
                if dist < min_dist:
                    min_dist = dist
                    closest_line = line
                    closest_x = x_val
                    closest_y = y_val

            if closest_line is not None:
                if annotation:
                    annotation.remove()

                roi_name = closest_line.get_label()
                annotation = ax.annotate(
                    roi_name,
                    xy=(closest_x, closest_y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    color=closest_line.get_color(),  # Same color as the trace
                    bbox=None  # No background box
                )
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()
        
def setup_plot_and_interaction(
    app, smoothed_dff_df, peak_results_df, fs,
    display_window, y_ax_range,
    export_dir, file_prefix,
    data_loader
):
    ca_marker = calcium_trace_symbol()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    plt.subplots_adjust(hspace=0.4, left=0.07, right=0.79)

    plotter = InteractivePlotter(
        smoothed_dff_df, peak_results_df, fs,
        display_window, y_ax_range, ca_marker,
        fig=fig, ax1=ax1, ax2=ax2
    )

    analyzer = InteractionHandler(
        data_loader=data_loader,
        plotter=plotter,
        peak_results_df=peak_results_df,
        smoothed_dff_df=smoothed_dff_df,
        fs=fs,
        display_window=display_window,
        y_ax_range=y_ax_range,
        ca_marker=ca_marker,
        directory=export_dir,
        fig=fig,
        ax1=ax1,
        ax2=ax2,
        file_prefix=file_prefix,
        pre_range_input=app.pre_range_input,
        post_range_input=app.post_range_input,
        truncate_sec=float(app.truncate_seconds_input.text()),
        polyorder=int(app.poly_order_input.text()),
        window_length_sec=float(app.smoothing_window_input.text())
    )

    analyzer.setup_interactive_plot()
    analyzer.setup_buttons()
    analyzer.fig.canvas.mpl_connect('key_press_event', analyzer.on_key)
    analyzer.update_plot()

    return analyzer.fig
