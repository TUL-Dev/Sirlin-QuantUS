import os
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt6.uic.load_ui import loadUi

from quantus.image_loading.utc_loaders.options import get_scan_loaders
from quantus.data_objs import UltrasoundRfImage
from quantus.entrypoints import scan_loading_step


def select_image_helper(path_input, file_exts):
    file_exts = " ".join([f"*{ext}" for ext in file_exts])
    if not os.path.exists(path_input.text()):  # check if file path is manually typed
        file_name, _ = QFileDialog.getOpenFileName(None, "Open File", filter=file_exts)
        if file_name != "":  # If valid file is chosen
            path_input.setText(file_name)
        else:
            return
        
class ScanLoadingWorker(QThread):
    """Worker thread for time-consuming operations."""
    finished = pyqtSignal(UltrasoundRfImage)  # Signal when process is done
    error_msg = pyqtSignal(str)  # Signal for error messages

    def __init__(self, scan_type, image_path, phantom_path, scan_loader_kwargs):
        super().__init__()
        self.scan_type = scan_type
        self.image_path = image_path
        self.phantom_path = phantom_path
        self.scan_loader_kwargs = scan_loader_kwargs
        
    def run(self):
        try:
            image_data = scan_loading_step(self.scan_type, self.image_path, self.phantom_path, **self.scan_loader_kwargs)
        except Exception as e:
            self.error_msg.emit(f"Error loading image: {e}")
            self.finished.emit(UltrasoundRfImage("",""))
            return
        
        # Emit the finished signal when done
        self.finished.emit(image_data)

from .select_image_ui import Ui_selectImage
class SelectImageGUI(QDialog, Ui_selectImage):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # super(SelectImageGUI, self).__init__()
        # loadUi(str(Path("quantus/gui/image_loading/select_image.ui")), self)
        
        self.scan_type_dropdown.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setLayout(self.full_screen_layout)
        self.full_screen_layout.removeItem(self.img_selection_layout)
        self.hide_img_selection_layout()
        self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
        self.full_screen_layout.setStretchFactor(self.select_type_layout, 10)

        self.scan_type = None
        self.file_exts = None
        self.image_data = None
        self.scan_loading_worker = None
        
        self.accept_type_button.clicked.connect(self.type_accepted)
        self.choose_image_path_button.clicked.connect(self.select_image_helper)
        self.choose_phantom_path_button.clicked.connect(self.selectPhantomFile)
        self.clear_image_path_button.clicked.connect(self.image_path_input.clear)
        self.clear_phantom_path_button.clicked.connect(self.phantom_path_input.clear)
        self.generate_image_button.clicked.connect(self.generate_image)
        self.back_button.clicked.connect(self.back_to_type_selection)
        
        self.scan_loaders = get_scan_loaders()
        scan_loaders = [s.replace("_", " ").capitalize() for s in self.scan_loaders.keys()]
        self.scan_loaders_list = [s.replace("rf", "RF").replace("iq", "IQ") for s in scan_loaders]
        self.scan_type_dropdown.addItems(self.scan_loaders_list)
        
    def hide_type_selection_layout(self):
        self.select_type_label.hide()
        self.scan_type_dropdown.hide()
        self.accept_type_button.hide()
        
    def show_type_selection_layout(self):
        self.select_type_label.show()
        self.scan_type_dropdown.show()
        self.accept_type_button.show()
        
    def hide_img_selection_layout(self):
        self.generate_image_button.hide()
        self.choose_phantom_path_button.hide()
        self.clear_phantom_path_button.hide()
        self.phantom_path_input.hide()
        self.phantom_path_label.hide()
        self.choose_image_path_button.hide()
        self.clear_image_path_button.hide()
        self.image_path_input.hide()
        self.image_path_label.hide()
        self.select_data_label.hide()
        self.select_image_error_msg.hide()
        self.analysis_kwargs_box.hide()
        self.analysis_kwargs_label.hide()
        self.back_button.hide()
        self.loading_screen_label.hide()
    
    def show_img_selection_layout(self):
        self.generate_image_button.show()
        self.choose_phantom_path_button.show()
        self.clear_phantom_path_button.show()
        self.phantom_path_input.show()
        self.phantom_path_label.show()
        self.choose_image_path_button.show()
        self.clear_image_path_button.show()
        self.image_path_input.show()
        self.image_path_label.show()
        self.select_data_label.show()
        self.analysis_kwargs_box.show()
        self.analysis_kwargs_label.show()
        self.back_button.show()
        
    def back_to_type_selection(self):
        self.show_type_selection_layout()
        self.hide_img_selection_layout()
        self.full_screen_layout.removeItem(self.img_selection_layout)
        self.full_screen_layout.addLayout(self.select_type_layout)
        self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
        self.full_screen_layout.setStretchFactor(self.select_type_layout, 10)
        self.select_data_label.setText("Select Scan Type")
        
        # Reset the image and phantom paths
        self.image_path_input.clear()
        self.phantom_path_input.clear()
        
    def select_image_helper(self):
        select_image_helper(self.image_path_input, self.file_exts)
        self.select_image_error_msg.hide()

    def selectPhantomFile(self):
        select_image_helper(self.phantom_path_input, self.file_exts)
        self.select_image_error_msg.hide()
        
    def type_accepted(self):
        self.scan_type = list(self.scan_loaders.keys())[self.scan_type_dropdown.currentIndex()]
        self.file_exts = self.scan_loaders[self.scan_type]["file_exts"]
        self.image_path_label.setText(f"Input Path to Image file\n ({', '.join(self.file_exts)})")
        self.phantom_path_label.setText(f"Input Path to Phantom file\n ({', '.join(self.file_exts)})")
        
        self.hide_type_selection_layout()
        self.full_screen_layout.removeItem(self.select_type_layout)
        self.full_screen_layout.addLayout(self.img_selection_layout)
        self.show_img_selection_layout()
        self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
        self.full_screen_layout.setStretchFactor(self.img_selection_layout, 10)
        self.select_data_label.setText(f"Select {self.scan_type_dropdown.currentText()} Image and Phantom files")
        
    def generate_image(self):
        kwargs_text = self.analysis_kwargs_box.toPlainText()
        if kwargs_text == "":
            kwargs_text = "{\n}"
        try:
            scan_loader_kwargs = eval(kwargs_text)
            assert isinstance(scan_loader_kwargs, dict), "Options must be in JSON format."
        except SyntaxError:
            self.select_image_error_msg.setText("Invalid options format. Please use a JSON format.")
            self.select_image_error_msg.show()
            return
        except Exception as e:
            self.select_image_error_msg.setText(f"Error in options: {e}")
            self.select_image_error_msg.show()
            return
        
        self.loading_screen_label.show()
        self.generate_image_button.hide()
        
        self.scan_loading_worker = ScanLoadingWorker(self.scan_type, self.image_path_input.text(), self.phantom_path_input.text(), scan_loader_kwargs)
        self.scan_loading_worker.error_msg.connect(self.select_image_error_msg.setText)
        self.scan_loading_worker.finished.connect(self.generate_image_complete)
        self.scan_loading_worker.start()
        
    def generate_image_complete(self, image_data: UltrasoundRfImage):
        """This function is called when the image generation is complete."""
        self.loading_screen_label.hide()
        self.generate_image_button.show()
        if image_data.scan_path == "":
            self.select_image_error_msg.show()
            return
        self.image_data = image_data
        
    def closeEvent(self, event):
        # Wait for thread to finish if it's running
        if self.scan_loading_worker is not None and self.scan_loading_worker.isRunning():
            self.scan_loading_worker.wait()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    ui = SelectImageGUI()
    ui.show()
    sys.exit(app.exec())
