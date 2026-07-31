from PyQt5.QtWidgets import (
    QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox,
    QListWidget, QAbstractItemView
)

def build_ui(parent):
    """
    Builds and returns the UI layout and a dictionary of UI elements.

    Parameters:
        parent: The QWidget or QApplication parent for the UI components.

    Returns:
        layout: The constructed QVBoxLayout.
        ui_elements: A dictionary of references to individual UI widgets.
    """
    layout = QVBoxLayout()
    ui_elements = {}

    # File selection
    ui_elements['file_list_widget'] = QListWidget()
    ui_elements['file_list_widget'].setSelectionMode(QAbstractItemView.SingleSelection)
    layout.addWidget(QLabel("Select recording to process:"))
    layout.addWidget(ui_elements['file_list_widget'])

    # File type dropdown
    layout.addWidget(QLabel("Select file type:"))
    ui_elements['filetype_combo'] = QComboBox()
    ui_elements['filetype_combo'].addItems(["CSV", "NPY"])
    layout.addWidget(ui_elements['filetype_combo'])

    ui_elements['file_button'] = QPushButton("Select Folder")
    layout.addWidget(ui_elements['file_button'])

    ui_elements['file_label'] = QLabel("No folder selected")
    layout.addWidget(ui_elements['file_label'])

    # Checkbox for filtering Suite2p cells (hidden by default)
    ui_elements['cell_filter_checkbox'] = QCheckBox("REMOVE NON-CELL suite2p ROIs")
    ui_elements['cell_filter_checkbox'].setChecked(True)
    ui_elements['cell_filter_checkbox'].setVisible(False)
    # We'll insert this above the NPY filename input later

    # NPY filename input
    ui_elements['npy_filename_input'] = QLineEdit("F.npy")
    ui_elements['npy_filename_input'].setPlaceholderText("Enter NPY filename (e.g., data.npy)")
    ui_elements['npy_filename_input'].setEnabled(False)
    layout.addWidget(QLabel("NPY Filename (only if NPY selected):"))
    layout.addWidget(ui_elements['npy_filename_input'])

    # Insert checkbox above NPY input
    layout.insertWidget(layout.indexOf(ui_elements['npy_filename_input']), ui_elements['cell_filter_checkbox'])

    # Other analysis parameters
    layout.addWidget(QLabel("Sampling frequency (Hz):"))
    ui_elements['fs_input'] = QLineEdit("5")
    layout.addWidget(ui_elements['fs_input'])

    layout.addWidget(QLabel("Truncate beginning (sec):"))
    ui_elements['truncate_seconds_input'] = QLineEdit("0")
    layout.addWidget(ui_elements['truncate_seconds_input'])

    layout.addWidget(QLabel("Prominence (float):"))
    ui_elements['prominence_input'] = QLineEdit("0.3")
    layout.addWidget(ui_elements['prominence_input'])

    layout.addWidget(QLabel("Minimum Peak Height (float, optional):"))
    ui_elements['min_height_input'] = QLineEdit()
    ui_elements['min_height_input'].setPlaceholderText("Leave blank to disable")
    layout.addWidget(ui_elements['min_height_input'])

    layout.addWidget(QLabel("Minimum Plateau Size (sec, optional):"))
    ui_elements['min_plateau_input'] = QLineEdit()
    ui_elements['min_plateau_input'].setPlaceholderText("Leave blank to disable")
    layout.addWidget(ui_elements['min_plateau_input'])

    layout.addWidget(QLabel("Polynomial Order (int):"))
    ui_elements['poly_order_input'] = QLineEdit("3")
    layout.addWidget(ui_elements['poly_order_input'])

    layout.addWidget(QLabel("Smoothing Window Length (sec):"))
    ui_elements['smoothing_window_input'] = QLineEdit("6")
    layout.addWidget(ui_elements['smoothing_window_input'])

    # Toggle for the Savitzky-Golay smoothing filter (applied by default).
    # When unchecked, the raw ΔF/F is used without smoothing.
    ui_elements['savgol_checkbox'] = QCheckBox("Apply Savitzky-Golay smoothing filter (delete artifact removal files first)")
    ui_elements['savgol_checkbox'].setChecked(True)
    layout.addWidget(ui_elements['savgol_checkbox'])

    # --------------------------------------------------------------
    # ΔF/F baseline (F0) method
    # --------------------------------------------------------------
    # "Whole-trace mean" is the original Calcify behavior. "Rolling percentile"
    # reproduces the standalone drift-correction script: a time-varying F0 that
    # rides under the transients and tracks slow drift (photobleaching/z-drift),
    # which also changes the amplitude scale to match that script.
    layout.addWidget(QLabel("ΔF/F baseline (F0) method:"))
    ui_elements['baseline_method_combo'] = QComboBox()
    ui_elements['baseline_method_combo'].addItems(
        ["Whole-trace mean", "Rolling percentile (drift correction)"]
    )
    layout.addWidget(ui_elements['baseline_method_combo'])

    layout.addWidget(QLabel("Rolling baseline window (sec):"))
    ui_elements['baseline_window_input'] = QLineEdit("60")
    layout.addWidget(ui_elements['baseline_window_input'])

    layout.addWidget(QLabel("Rolling baseline percentile (0-100):"))
    ui_elements['baseline_pctl_input'] = QLineEdit("10")
    layout.addWidget(ui_elements['baseline_pctl_input'])

    # --------------------------------------------------------------
    # Neuropil subtraction (Suite2p NPY only): F - coef * Fneu
    # --------------------------------------------------------------
    ui_elements['neuropil_checkbox'] = QCheckBox(
        "Subtract neuropil (F - coef*Fneu; Suite2p NPY only)"
    )
    ui_elements['neuropil_checkbox'].setChecked(False)
    layout.addWidget(ui_elements['neuropil_checkbox'])

    layout.addWidget(QLabel("Neuropil coefficient:"))
    ui_elements['neuropil_coef_input'] = QLineEdit("0.7")
    layout.addWidget(ui_elements['neuropil_coef_input'])

    layout.addWidget(QLabel("Zoomed In Display Window (sec):"))
    ui_elements['display_window_input'] = QLineEdit("200")
    layout.addWidget(ui_elements['display_window_input'])

    layout.addWidget(QLabel("Y-axis range (comma-separated floats, e.g. -1,1):"))
    ui_elements['yax_input'] = QLineEdit("-1,1")
    layout.addWidget(ui_elements['yax_input'])

    ui_elements['transpose_checkbox'] = QCheckBox("Transpose data? (Ignore for Suite2p)")
    layout.addWidget(ui_elements['transpose_checkbox'])

    # Optional pre/post range inputs
    ui_elements['pre_range_input'] = QLineEdit()
    ui_elements['pre_range_input'].setPlaceholderText("Optional: Pre Range (e.g., 0-100)")
    layout.addWidget(ui_elements['pre_range_input'])
    
    ui_elements['post_range_input'] = QLineEdit()
    ui_elements['post_range_input'].setPlaceholderText("Optional: Post Range (e.g., 200-300)")
    layout.addWidget(ui_elements['post_range_input'])

    # Swap the order of the buttons: Plot All ROIs first, then Run Analysis
    ui_elements['plot_all_button'] = QPushButton("Plot All ROIs")
    layout.addWidget(ui_elements['plot_all_button'])

    ui_elements['run_button'] = QPushButton("Run Analysis")
    layout.addWidget(ui_elements['run_button'])

    # Settings persistence: save the current parameter fields to a
    # calcify_config.json (typed, no extra dependencies) and load them back.
    ui_elements['save_settings_button'] = QPushButton("Save Settings")
    layout.addWidget(ui_elements['save_settings_button'])

    ui_elements['load_settings_button'] = QPushButton("Load Settings")
    layout.addWidget(ui_elements['load_settings_button'])

    # Return the layout and UI elements dictionary
    return layout, ui_elements
