from pathlib import Path

import numpy as np

class UltrasoundRfImage:
    """
    Class for ultrasound RF image data.
    """

    def __init__(self, scan_path: str, phantom_path: str):
        # RF data
        self.rf_data: np.ndarray
        self.phantom_rf_data: np.ndarray
        self.bmode: np.ndarray
        self.axial_res: float # mm/pix
        self.lateral_res: float # mm/pix
        self.scan_name = Path(scan_path).stem
        self.phantom_name = Path(phantom_path).stem
        self.scan_path = scan_path
        self.phantom_path = phantom_path
        self.spatial_dims: int
        
        # Scan conversion parameters
        self.sc_bmode: np.ndarray = None
        self.xmap: np.ndarray # sc (y,x) --> preSC x
        self.ymap: np.ndarray # sc (y,x) --> preSC y
        self.width: float # deg
        self.tilt: float
        self.start_depth: float # mm
        self.end_depth: float # mm
        self.sc_axial_res: float # mm
        self.sc_lateral_res: float
        
        # 3D scan parameters
        self.coronal_res: float = None # mm/pix
        self.depth: float # depth in mm
        
        # 3D scan conversion parameters
        self.sc_bmode: np.ndarray = None
        self.coord_map_3d: np.ndarray # maps (z,y,x) in SC coords to (x,y) preSC coord
        self.sc_axial_res: float # mm/pix
        self.sc_lateral_res: float # mm/pix
        self.sc_coronal_res: float # mm/pix
        self.sc_params_3d = None
