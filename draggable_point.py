import numpy as np

class DraggablePoint:
    def __init__(self, ax, x, y, color, label, time_axis, trace, fs, smoothed_dff_df, current_roi, plotter, role=None):
        self.ax = ax
        self.x = x
        self.y = y
        self.color = color
        self.label = label
        self.plot, = ax.plot([x], [y], 'o', color=color, label=label, picker=True)
        self.cid = self.plot.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.dragging = False
        self.fs = fs
        self.smoothed_dff_df = smoothed_dff_df
        self.current_roi = current_roi
        self.plotter = plotter
        self.time_axis = time_axis
        self.role = role

    def _get_master_row_label(self):
        """
        Resolve the index label in peak_results_df for the currently displayed peak.

        peak_idx is a position within peaks_in_roi (the ROI-filtered slice).
        We look up that row's peak_time and cell_id, then find the matching
        row in the master peak_results_df to get a stable label.
        """
        peak_results_df = self.plotter.peak_results_df
        peaks_in_roi = self.plotter.peaks_in_roi
        peak_pos = self.plotter.peak_idx

        # Get the identifying values from the ROI slice
        roi_row = peaks_in_roi.iloc[peak_pos]
        cell_id = roi_row['cell_id']
        peak_time = roi_row['peak_time']

        # Find the matching row in the master DataFrame
        mask = (
            (peak_results_df['cell_id'] == cell_id) &
            np.isclose(peak_results_df['peak_time'], peak_time, atol=0.5 / self.fs)
        )
        matching = peak_results_df.index[mask]

        if len(matching) == 0:
            raise KeyError(
                f"Could not find peak in master DataFrame: cell_id={cell_id}, peak_time={peak_time}"
            )

        return matching[0]

    def on_pick(self, event):
        if event.artist != self.plot:
            return
        self.dragging = True
        # Disconnect previous motion/release handlers if they exist
        if hasattr(self, '_motion_cid'):
            self.plot.figure.canvas.mpl_disconnect(self._motion_cid)
        if hasattr(self, '_release_cid'):
            self.plot.figure.canvas.mpl_disconnect(self._release_cid)
        self._motion_cid  = self.plot.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self._release_cid = self.plot.figure.canvas.mpl_connect('button_release_event', self.on_release)
        
    def on_motion(self, event):
        if self.dragging and event.xdata is not None:
            # Snap to nearest sample
            self.x = np.clip(event.xdata, self.time_axis[0], self.time_axis[-1])
            time_idx = np.argmin(np.abs(self.time_axis - self.x))
            self.x = self.time_axis[time_idx]
            # Update y to follow the trace
            self.update_base_value()
            # Redraw the marker only; skip AUC shading during drag for performance
            self.plot.set_data([self.x], [self.y])
            self.ax.figure.canvas.draw_idle()
        
    def update_auc_shading(self):
        # Remove old patch
        if hasattr(self, 'auc_patch') and self.auc_patch in self.ax.patches:
            self.auc_patch.remove()

        peak_results_df = self.plotter.peak_results_df

        # Safely resolve which master row we're editing
        try:
            row_label = self._get_master_row_label()
        except KeyError as e:
            print(f"[update_auc_shading] {e}")
            return

        left_base_time  = peak_results_df.loc[row_label, 'left_bases']
        right_base_time = peak_results_df.loc[row_label, 'right_bases']

        if np.isnan(left_base_time) or np.isnan(right_base_time):
            return

        left_idx  = np.searchsorted(self.time_axis, left_base_time)
        right_idx = np.searchsorted(self.time_axis, right_base_time)

        trace = self.smoothed_dff_df.loc[self.current_roi].values
        time_axis = np.arange(len(trace)) / self.fs

        left_base_val  = np.interp(left_base_time, time_axis, trace)
        right_base_val = np.interp(right_base_time, time_axis, trace)
        base_val = min(left_base_val, right_base_val)

        self.auc_patch = self.ax.fill_between(
            time_axis[left_idx:right_idx + 1],
            base_val,
            trace[left_idx:right_idx + 1],
            color='purple',
            alpha=0.3
        )
        self.ax.figure.canvas.draw_idle()

    def on_release(self, event):
        self.dragging = False
    
        peak_results_df = self.plotter.peak_results_df
    
        try:
            row_label = self._get_master_row_label()
        except KeyError as e:
            print(f"[on_release] {e}")
            return
    
        # --- Step 1: Write new base position ---
        if self.role == 'left_base':
            peak_results_df.loc[row_label, 'left_bases'] = self.x
        elif self.role == 'right_base':
            peak_results_df.loc[row_label, 'right_bases'] = self.x
    
        # --- Step 2: Recompute peak metrics ---
        left_base_time  = peak_results_df.loc[row_label, 'left_bases']
        right_base_time = peak_results_df.loc[row_label, 'right_bases']
        trace = self.smoothed_dff_df.loc[self.current_roi].values
        time_axis = np.arange(len(trace)) / self.fs
    
        left_base_val  = np.interp(left_base_time, time_axis, trace)
        right_base_val = np.interp(right_base_time, time_axis, trace)
        base_value = min(left_base_val, right_base_val)
        peak_results_df.loc[row_label, 'base_value'] = base_value
    
        left_idx  = int(round(left_base_time * self.fs))
        right_idx = int(round(right_base_time * self.fs))
        window_trace = trace[left_idx:right_idx + 1]
        window_time  = time_axis[left_idx:right_idx + 1]
    
        peak_rel_idx = np.argmax(window_trace)
        peak_time    = window_time[peak_rel_idx]
        peak_value   = window_trace[peak_rel_idx]
        peak_results_df.loc[row_label, 'peak_time']  = peak_time
        peak_results_df.loc[row_label, 'peak_value'] = peak_value
        peak_results_df.loc[row_label, 'prominences'] = peak_value - base_value
    
        half_rise_val = base_value + (peak_value - base_value) / 2
        try:
            rel_idx = np.where(window_trace >= half_rise_val)[0][0]
            half_rise_time = window_time[rel_idx] - left_base_time
        except IndexError:
            half_rise_time = np.nan
        peak_results_df.loc[row_label, 'half_rise_time'] = half_rise_time
    
        decay_trace     = window_trace[peak_rel_idx:]
        decay_time_axis = window_time[peak_rel_idx:]
        try:
            rel_idx = np.where(decay_trace <= half_rise_val)[0][0]
            half_decay_time = decay_time_axis[rel_idx] - peak_time
        except IndexError:
            half_decay_time = np.nan
        peak_results_df.loc[row_label, 'half_decay_time'] = half_decay_time
    
        # --- Step 3: Update marker vertical position ---
        self.update_base_value()
    
        # --- Step 4: Sort once at the end, before refreshing peaks_in_roi ---
        self.plotter.peak_results_df = peak_results_df.sort_values(
            ['cell_id', 'peak_time']
        ).reset_index(drop=True)
    
        # --- Step 5: Refresh peaks_in_roi using the now-sorted df ---
        self.plotter.peaks_in_roi = self.plotter._get_peaks_for_current_roi()
    
        # --- Step 6: Replot ---
        # Re-resolve row_label after the sort, since the index has changed
        updated_peak_row = self.plotter.peak_results_df[
            (self.plotter.peak_results_df['cell_id'] == self.current_roi) &
            np.isclose(self.plotter.peak_results_df['peak_time'], peak_time, atol=0.5 / self.fs)
        ].iloc[0]
    
        self.plotter._plot_peak_region(updated_peak_row, trace, time_axis)
    
        # --- Step 7: Update AUC shading ---
        self.update_auc_shading()
    
        print(f"[DEBUG draggable] Stored → left={left_base_time:.6f}s, right={right_base_time:.6f}s, "
              f"base={base_value:.6f}, peak={peak_value:.6f}, "
              f"prominence={peak_results_df.loc[row_label, 'prominences']:.6f}, "
              f"half-rise={half_rise_time}, half-decay={half_decay_time}")

    def update_plot(self):
        self.plot.set_data([self.x], [self.y])
        self.ax.figure.canvas.draw_idle()

    def update_base_value(self):
        trace = self.smoothed_dff_df.loc[self.current_roi].values
        time_axis = np.arange(len(trace)) / self.fs
        self.y = np.interp(self.x, time_axis, trace)
        self.y = np.clip(self.y, trace.min(), trace.max())

    def update_data(self):
        """Legacy method, retained for backward compatibility."""
        peak_results_df = self.plotter.peak_results_df

        try:
            row_label = self._get_master_row_label()
        except KeyError as e:
            print(f"[update_data] {e}")
            return

        # Write base position
        if self.label == 'Left Base':
            peak_results_df.loc[row_label, 'left_bases'] = self.x
        elif self.label == 'Right Base':
            peak_results_df.loc[row_label, 'right_bases'] = self.x

        # Refresh ROI slice
        self.plotter.peaks_in_roi = peak_results_df[
            peak_results_df['cell_id'] == self.current_roi
        ].copy().reset_index(drop=True)

        left_base_time  = peak_results_df.loc[row_label, 'left_bases']
        right_base_time = peak_results_df.loc[row_label, 'right_bases']

        if np.isnan(left_base_time) or np.isnan(right_base_time):
            print("Invalid base window selected.")
            return

        left_idx  = int(left_base_time * self.fs)
        right_idx = int(right_base_time * self.fs)
        trace = self.smoothed_dff_df.loc[self.current_roi].values

        if left_idx >= right_idx or left_idx < 0 or right_idx >= len(trace):
            print("Invalid base window selected.")
            return

        window_trace = trace[left_idx:right_idx + 1]
        window_time  = np.arange(left_idx, right_idx + 1) / self.fs

        new_peak_rel   = np.argmax(window_trace)
        new_peak_time  = window_time[new_peak_rel]
        new_peak_value = window_trace[new_peak_rel]

        if self.label == 'Left Base':
            peak_results_df.loc[row_label, 'left_base_value'] = np.interp(
                left_base_time, window_time, window_trace)
        elif self.label == 'Right Base':
            peak_results_df.loc[row_label, 'right_base_value'] = np.interp(
                right_base_time, window_time, window_trace)

        new_base_value = min(
            peak_results_df.loc[row_label, 'left_base_value'],
            peak_results_df.loc[row_label, 'right_base_value']
        )

        peak_results_df.loc[row_label, 'base_value']  = new_base_value
        peak_results_df.loc[row_label, 'peak_time']   = new_peak_time
        peak_results_df.loc[row_label, 'peak_value']  = new_peak_value
        peak_results_df.loc[row_label, 'prominences'] = new_peak_value - new_base_value

        self.plotter.update_plot()