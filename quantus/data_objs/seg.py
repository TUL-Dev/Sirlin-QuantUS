import numpy as np
from typing import List

class BmodeSeg:
    """
    Class for ultrasound RF image data.
    """

    def __init__(self):
        self.seg_name: str
        self.splines: List[np.ndarray] # [X, Y, (Z)]
        self.seg_mask: np.ndarray
        self.sc_seg_mask: np.ndarray
        self.frame: int
        
        # Scan conversion parameters
        self.sc_splines: List[np.ndarray] # [X, Y, (Z)]
