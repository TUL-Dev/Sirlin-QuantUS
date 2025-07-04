import pickle
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing

from .transforms import map_1d_to_3d
from .decorators import extensions
from ..data_objs.seg import BmodeSeg
from ..data_objs.image import UltrasoundRfImage

@extensions(".pkl", ".pickle")
def pkl_roi(image_data: UltrasoundRfImage, seg_path: str, **kwargs) -> BmodeSeg:
    """
    Function for loading ROI data from a pickle file saved from the QuantUS UI.
    All ROI data is exported in the same coordinates as the RF data.
    
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the ROI file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the ROI file name.
    """
    out = BmodeSeg()
    
    with open(seg_path, "rb") as f:
        roi_info: dict = pickle.load(f)
    
    if kwargs.get("assert_scan"):
        assert roi_info["Image Name"] == Path(kwargs["scan_path"]).name, 'Scan file name mismatch'
    if kwargs.get("assert_phantom"):
        assert roi_info["Phantom Name"] == Path(kwargs["phantom_path"]).name, 'Phantom file name mismatch'
    
    spline_x = np.array(roi_info["Spline X"])
    spline_y = np.array(roi_info["Spline Y"])
    if image_data.sc_bmode is not None:
        # Define the ROI spline on the same coordinates as the RF data
        rf_spline_x = np.array([image_data.xmap[int(y), int(x)] for x, y in zip(spline_x, spline_y)])
        rf_spline_y = np.array([image_data.ymap[int(y), int(x)] for x, y in zip(spline_x, spline_y)])
        out.sc_splines = [spline_x, spline_y]
        spline_x = rf_spline_x; spline_y = rf_spline_y
    
    spline = [(spline_x[i], spline_y[i]) for i in range(len(spline_x))]
    mask = Image.new("L", (image_data.phantom_rf_data.shape[1], image_data.phantom_rf_data.shape[0]), 0)
    ImageDraw.Draw(mask).polygon(spline, outline=1, fill=1)

    out.seg_mask = np.array(mask)
    out.splines = [spline_x, spline_y]
    out.frame = roi_info.get("Frame", 0)
    out.seg_name = Path(seg_path).stem
    
    return out
    
@extensions(".nii", ".nii.gz")
def nifti_voi(image_data: UltrasoundRfImage, seg_path: str, **kwargs) -> BmodeSeg:
    """
    Function for loading ROI data from a pickle file saved from the QuantUS UI.
     
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the ROI file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the ROI file name.
    """
    out = BmodeSeg()
    
    seg = nib.load(seg_path)
    out.sc_seg_mask = np.asarray(seg.dataobj, dtype=np.uint8)
    out.frame = 0
    out.seg_name = Path(seg_path).stem

    coord_map = image_data.coord_map_3d
    masked_coords_3d = np.transpose(np.where(out.sc_seg_mask))
    original_coords_3d = [map_1d_to_3d(coord_map[*coord], image_data.rf_data.shape[0], image_data.rf_data.shape[1], image_data.rf_data.shape[2])
                           for coord in masked_coords_3d]
    
    out.seg_mask = np.zeros_like(image_data.bmode)
    out.seg_mask[tuple(np.transpose(original_coords_3d))] = 1
    out.seg_mask = binary_closing(out.seg_mask, structure=np.ones((3, 3, 3))).astype(np.uint8)

    return out
