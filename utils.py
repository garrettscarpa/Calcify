def parse_analysis_parameters(app):
    try:
        fs = float(app.fs_input.text())
        truncate_seconds = float(app.truncate_seconds_input.text())
        truncate_samples = int(truncate_seconds * fs)
        prominence = float(app.prominence_input.text())

        min_height_text = app.min_height_input.text().strip()
        min_height = float(min_height_text) if min_height_text else None

        min_plateau_text = app.min_plateau_input.text().strip()
        min_plateau_samples = int(round(float(min_plateau_text) * fs)) if min_plateau_text else None

        y_ax_range = [float(x.strip()) for x in app.yax_input.text().split(",")]
        if len(y_ax_range) != 2:
            raise ValueError("Y-axis range must have two values")

        polynomial_order = int(app.poly_order_input.text())
        smoothing_window_sec = float(app.smoothing_window_input.text())
        smoothing_window_length = int(round(smoothing_window_sec * fs))
        if smoothing_window_length % 2 == 0:
            smoothing_window_length += 1

        display_window = int(round(float(app.display_window_input.text()) * fs))

        return {
            "fs": fs,
            "truncate_samples": truncate_samples,
            "prominence": prominence,
            "min_height": min_height,
            "min_plateau_samples": min_plateau_samples,
            "y_ax_range": y_ax_range,
            "polynomial_order": polynomial_order,
            "smoothing_window_length": smoothing_window_length,
            "display_window": display_window
        }
    except Exception as e:
        return None
