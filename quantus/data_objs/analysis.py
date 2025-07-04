from abc import ABC, abstractmethod

import numpy as np
from typing import List
from tqdm import tqdm

from .seg import BmodeSeg
from .image import UltrasoundRfImage
from .analysis_config import RfAnalysisConfig

class BlankResults:
    """To be filled dynamically with analysis results."""
    pass

class Window:
    """Class to store window data for sliding window analysis.
    """
    def __init__(self):
        self.lat_min = 0
        self.lat_max = 0
        self.ax_min = 0
        self.ax_max = 0
        self.cor_max = -1
        self.cor_min = -1
        self.results = BlankResults()
        
class ParamapAnalysisBase(ABC):
    """Facilitate parametric map-centric analysis of ultrasound images.
    """
    
    def __init__(self):
        self.image_data: UltrasoundRfImage
        self.config: RfAnalysisConfig
        self.seg_data: BmodeSeg
        
        self.windows: List[Window] = []
        self.single_window: Window
        self.spline_x: np.ndarray
        self.spline_y: np.ndarray
        self.spline_z: np.ndarray

    @abstractmethod
    def generate_seg_windows(self):
        """Generate windows for parametric map analysis based on user-defined segmentation."""
        pass
                    
    @abstractmethod
    def compute_window_vals(self, window: Window):
        """Compute parametric map values for a single window.
        
        Args:
            window (Window): Window object to store results.
        """
        pass

    def compute_paramaps(self):
        """Compute UTC parameters for each window in the ROI, creating a parametric map.
        """
        if not len(self.windows):
            self.generate_seg_windows()
            assert len(self.windows) > 0, "No windows generated"

        for window in tqdm(self.windows):
            self.compute_window_vals(window)
    
    @abstractmethod
    def compute_single_window(self):
        """Define a single window that contains all ROIs for analysis.
        """
        pass
