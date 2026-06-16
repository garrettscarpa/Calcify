import os
import glob
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem, QMessageBox

def filetype_changed(app):
    file_type = app.filetype_combo.currentText().upper()

    if file_type == "NPY":
        # Enable and show NPY filename input
        app.npy_filename_input.setEnabled(True)
        app.npy_filename_input.setVisible(True)

        # Show Suite2p cell filter checkbox
        app.cell_filter_checkbox.setVisible(True)
    else:
        # Disable and hide NPY filename input
        app.npy_filename_input.setEnabled(False)
        app.npy_filename_input.clear()
        app.npy_filename_input.setVisible(False)

        # Hide Suite2p cell filter checkbox
        app.cell_filter_checkbox.setVisible(False)

        # Reset folder paths and label
        app.npy_folder_path = None
        app.csv_folder_path = None
        app.file_label.setText("No file/folder selected")
        
def select_file_or_folder(app):
    file_type = app.filetype_combo.currentText()
    folder_path = QFileDialog.getExistingDirectory(app, "Select folder containing files")

    if folder_path:
        app.csv_folder_path = folder_path if file_type == "CSV" else None
        app.npy_folder_path = folder_path if file_type == "NPY" else None
        app.file_label.setText(f"Selected {file_type} folder: {folder_path}")
        populate_file_list(app)

def populate_file_list(app):
    app.file_list_widget.clear()
    file_type = app.filetype_combo.currentText()
    folder_path = app.csv_folder_path if file_type == "CSV" else app.npy_folder_path

    if not folder_path:
        return

    if file_type == "CSV":
        file_list = glob.glob(os.path.join(folder_path, '**', '*.csv'), recursive=True)

        # Exclude any CSVs that contain these substrings
        excluded_substrings = ['filtered_peaks', 'Individual_ROI_Statistics']
        file_list = [
            f for f in file_list
            if not any(sub in os.path.basename(f) for sub in excluded_substrings)
        ]
    else:
        npy_filename = app.npy_filename_input.text().strip()
        if not npy_filename:
            return
        file_list = glob.glob(os.path.join(folder_path, '**', npy_filename), recursive=True)

    for file_path in file_list:
        item = QListWidgetItem(file_path)
        app.file_list_widget.addItem(item)

def validate_and_get_file_list(app, selected_type):
    if selected_type == "CSV":
        folder = getattr(app, 'csv_folder_path', None)
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(app, "Error", "Please select a valid folder containing CSV files.")
            return None
        file_list = glob.glob(os.path.join(folder, '**', '*.csv'), recursive=True)
        if not file_list:
            QMessageBox.warning(app, "Error", "No CSV files found.")
            return None
        return file_list
    else:  # NPY
        folder = getattr(app, 'npy_folder_path', None)
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(app, "Error", "Please select a valid folder containing NPY files.")
            return None
        npy_filename = app.npy_filename_input.text().strip()
        if not npy_filename:
            QMessageBox.warning(app, "Error", "Please enter the NPY filename.")
            return None
        search_pattern = os.path.join(folder, '**', npy_filename)
        file_list = glob.glob(search_pattern, recursive=True)
        if not file_list:
            QMessageBox.warning(app, "Error", f"No file named '{npy_filename}' found.")
            return None
        return file_list

