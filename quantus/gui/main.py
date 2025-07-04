import sys

from PyQt6.QtWidgets import QStackedWidget, QApplication, QDialog

from quantus.gui.image_loading.select_image import SelectImageGUI
from quantus.gui.seg_loading.roi_selection import SegSelectionGUI as RoiGUI
from quantus.gui.seg_loading.voi_selection import SegSelectionGUI as VoiGUI

app = QApplication(sys.argv)
widget = QStackedWidget()
start_window = SelectImageGUI()

widget.setMinimumWidth(1400)
widget.addWidget(start_window)
widget.show()

def run_start_window(old_widget: QDialog = None):
    while start_window.image_data is None and start_window.isVisible():
        app.processEvents()
        if not start_window.isVisible():
            app.quit()
            sys.exit(0)
            
    if start_window.image_data.spatial_dims == 2:
        seg_window = RoiGUI(start_window.image_data)
    elif start_window.image_data.spatial_dims == 3:
        seg_window = VoiGUI(start_window.image_data)
    else:
        raise ValueError("Invalid spatial dimensions. Only 2D and 3D images are supported.")

    if old_widget is not None:
        widget.removeWidget(old_widget)
        old_widget.deleteLater()
    
    widget.addWidget(seg_window)
    widget.setCurrentWidget(seg_window)
    widget.show()
    run_seg_window(seg_window)
            
def run_seg_window(seg_window: RoiGUI | VoiGUI):
    while seg_window.isVisible():
        app.processEvents()
        if not seg_window.isVisible():
            app.quit()
            sys.exit(0)
        if seg_window.go_back:
            widget.setCurrentWidget(start_window)
            widget.show()
            start_window.image_data = None
            run_start_window()
            break

run_start_window()
