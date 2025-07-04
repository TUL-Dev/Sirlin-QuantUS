import numpy as np
from typing import List

from quantus.data_objs import UltrasoundRfImage, BmodeSeg, RfAnalysisConfig, ParamapAnalysisBase
from quantus.data_objs.analysis import Window
from .functions import *

class_name = "ParamapAnalysis"

class ParamapAnalysis(ParamapAnalysisBase):
    """
    Class to complete RF analysis via the sliding window technique
    and generate a corresponding parametric map.
    """
    def __init__(self, image_data: UltrasoundRfImage, config: RfAnalysisConfig, seg: BmodeSeg, 
                 function_names: List[str], **kwargs):
        # Type checking
        assert isinstance(image_data, UltrasoundRfImage), 'image_data must be an UltrasoundRfImage child class'
        assert isinstance(config, RfAnalysisConfig), 'config must be an RfAnalysisConfig'
        assert isinstance(seg, BmodeSeg), 'seg_data must be a BmodeSeg'
        super().__init__()
        
        self.analysis_kwargs = kwargs
        self.function_names = function_names
        self.seg_data = seg
        self.image_data = image_data
        self.config = config
        if hasattr(self.seg_data, "splines") and len(self.seg_data.splines) == 2:
            self.spline_x = seg.splines[0]
            self.spline_y = seg.splines[1]
        self.determine_func_order()
            
    def determine_func_order(self):
        """Determine the order of functions to be applied to the data.
        
        This function is called in the constructor and sets the order of functions
        to be applied to the data based on the provided function names.
        """
        self.ordered_funcs = []; self.ordered_func_names = []; self.results_names = []
        self.unordered_window_func_names = set(); self.unordered_full_seg_func_names = set()
        
        def assign_locs(func_name, deps, locs):
            """Assign locations for the function based on its dependencies and locations."""
            if 'window' in locs:
                self.unordered_window_func_names.add(func_name)
                [self.unordered_full_seg_func_names.add(dep) for dep in deps]
            if 'full_segmentation' in locs:
                self.unordered_full_seg_func_names.add(func_name)
                [self.unordered_window_func_names.add(dep) for dep in deps]
        
        def process_deps(func_name):
            if func_name in self.ordered_func_names:
                return
            if func_name in globals():
                # Handle function dependencies and outputs
                function = globals()[func_name]
                deps = function.get('deps', [])
                results_names = function.get('outputs', [])
                for dep in deps:
                    process_deps(dep)
                
                # Handle function locations
                locs = function.get('location', ['window', 'full_segmentation'])
                assign_locs(func_name, deps, locs)
            else:
                raise ValueError(f"Function '{func_name}' not found!")
            
            self.ordered_funcs.append(function['func'])
            self.ordered_func_names.append(func_name)
            self.results_names.extend(results_names)

        for function_name in self.function_names:
            process_deps(function_name)
            
    def generate_seg_windows_3d(self):
        """Generate 3D voxel windows for UTC analysis based on user-defined spline."""
        # Some axial/lateral/coronal dims
        axial_pix_size = round(self.config.ax_win_size / self.image_data.axial_res)  # mm/(mm/pix)
        lateral_pix_size = round(self.config.lat_win_size / self.image_data.lateral_res)  # mm(mm/pix)
        coronal_pix_size = round(self.config.cor_win_size / self.image_data.coronal_res)  # mm/(mm/pix)
        
        # Overlap fraction determines the incremental distance between windows
        axial_increment = axial_pix_size * (1 - self.config.axial_overlap)
        lateral_increment = lateral_pix_size * (1 - self.config.lateral_overlap)
        coronal_increment = coronal_pix_size * (1 - self.config.coronal_overlap)
        
        # Determine windows - Find Volume to Iterate Over
        axial_start = np.min(np.where(np.any(self.seg_data.seg_mask, axis=(0, 1)))[0])
        axial_end = np.max(np.where(np.any(self.seg_data.seg_mask, axis=(0, 1)))[0])
        lateral_start = np.min(np.where(np.any(self.seg_data.seg_mask, axis=(0, 2)))[0])
        lateral_end = np.max(np.where(np.any(self.seg_data.seg_mask, axis=(0, 2)))[0])
        coronal_start = np.min(np.where(np.any(self.seg_data.seg_mask, axis=(1, 2)))[0])
        coronal_end = np.max(np.where(np.any(self.seg_data.seg_mask, axis=(1, 2)))[0])
        
        self.windows = []
        
        for axial_pos in np.arange(axial_start, axial_end, axial_increment):
            for lateral_pos in np.arange(lateral_start, lateral_end, lateral_increment):
                for coronal_pos in np.arange(coronal_start, coronal_end, coronal_increment):
                    # Convert axial, lateral, and coronal positions to indices
                    axial_ind = np.round(axial_pos).astype(int)
                    lateral_ind = np.round(lateral_pos).astype(int)
                    coronal_ind = np.round(coronal_pos).astype(int)
                    
                    # Determine if window is inside analysis volume
                    mask_vals = self.seg_data.seg_mask[
                        coronal_ind : (coronal_ind + coronal_pix_size),
                        lateral_ind : (lateral_ind + lateral_pix_size),
                        axial_ind : (axial_ind + axial_pix_size),
                    ]
                    
                    # Define Percentage Threshold
                    total_number_of_elements_in_region = mask_vals.size
                    number_of_ones_in_region = len(np.where(mask_vals == True)[0])
                    percentage_ones = number_of_ones_in_region / total_number_of_elements_in_region
                    
                    if percentage_ones > self.config.window_thresh:
                        # Add ROI to output structure, quantize back to valid distances
                        new_window = Window()
                        new_window.ax_min = int(axial_pos)
                        new_window.ax_max = int(axial_pos + axial_pix_size)
                        new_window.lat_min = int(lateral_pos)
                        new_window.lat_max = int(lateral_pos + lateral_pix_size)
                        new_window.cor_min = int(coronal_pos)
                        new_window.cor_max = int(coronal_pos + coronal_pix_size)
                        self.windows.append(new_window)
        
    def generate_seg_windows(self):
        """Generate windows for parametric map analysis based on user-defined segmentation.
        """
        if len(self.seg_data.seg_mask.shape) == 3: # 3D analysis
            return self.generate_seg_windows_3d()
        
        # Some axial/lateral dims
        ax_pix_size = round(self.config.ax_win_size / self.image_data.axial_res)  # mm/(mm/pix)
        lat_pix_size = round(self.config.lat_win_size / self.image_data.lateral_res)  # mm/(mm/pix)
        
        axial = list(range(self.image_data.rf_data.shape[0]))
        lateral = list(range(self.image_data.rf_data.shape[1]))

        # Overlap fraction determines the incremental distance between ROIs
        axial_increment = ax_pix_size * (1 - self.config.axial_overlap)
        lateral_increment = lat_pix_size * (1 - self.config.lateral_overlap)

        # Determine ROIS - Find Region to Iterate Over
        axial_start = max(min(self.spline_y), axial[0])
        axial_end = min(max(self.spline_y), axial[-1] - ax_pix_size)
        lateral_start = max(min(self.spline_x), lateral[0])
        lateral_end = min(max(self.spline_x), lateral[-1] - lat_pix_size)

        self.windows = []
        mask = self.seg_data.seg_mask

        for axial_pos in np.arange(axial_start, axial_end, axial_increment):
            for lateral_pos in np.arange(lateral_start, lateral_end, lateral_increment):
                # Convert axial and lateral positions in mm to Indices
                axial_abs_ar = abs(axial - axial_pos)
                axial_ind = np.where(axial_abs_ar == min(axial_abs_ar))[0][0]
                lateral_abs_ar = abs(lateral - lateral_pos)
                lateral_ind = np.where(lateral_abs_ar == min(lateral_abs_ar))[0][0]

                # Determine if ROI is Inside Analysis Region
                mask_vals = mask[
                    axial_ind : (axial_ind + ax_pix_size),
                    lateral_ind : (lateral_ind + lat_pix_size),
                ]

                # Define Percentage Threshold
                total_elements_in_region = mask_vals.size
                ones_in_region = len(np.where(mask_vals == 1)[0])
                percentage_ones = ones_in_region / total_elements_in_region

                if percentage_ones > self.config.window_thresh:
                    # Add window to output structure, quantize back to valid distances
                    new_window = Window()
                    new_window.lat_min = int(lateral[lateral_ind])
                    new_window.lat_max = int(lateral[lateral_ind + lat_pix_size - 1])
                    new_window.ax_min = int(axial[axial_ind])
                    new_window.ax_max = int(axial[axial_ind + ax_pix_size - 1])
                    self.windows.append(new_window)

    def compute_window_vals(self, window):
        """Compute parametric map values for a single window.
        
        Args:
            window (Window): Window object to store results.
        """
        if self.image_data.bmode.ndim == 2:
            img_window = self.image_data.rf_data[
                window.ax_min: window.ax_max + 1, window.lat_min: window.lat_max + 1
            ]
            phantom_window = self.image_data.phantom_rf_data[
                window.ax_min: window.ax_max + 1, window.lat_min: window.lat_max + 1
            ]
        elif self.image_data.bmode.ndim == 3:
            img_window = self.image_data.rf_data[
                window.cor_min: window.cor_max + 1, window.lat_min: window.lat_max + 1,
                window.ax_min: window.ax_max + 1
            ]
            phantom_window = self.image_data.phantom_rf_data[
                window.cor_min: window.cor_max + 1, window.lat_min: window.lat_max + 1,
                window.ax_min: window.ax_max + 1
            ]
        else:
            raise ValueError("Invalid RF data dimensions. Expected 2D or 3D data.")

        for i, function in enumerate(self.ordered_funcs):
            if self.ordered_func_names[i] in self.unordered_window_func_names:
                function(img_window, phantom_window, window, self.config, self.image_data, **self.analysis_kwargs)

    def compute_single_window(self):
        """Define a single window that contains all ROIs for analysis and compute full signal functions.
        """
        min_ax = min([window.ax_min for window in self.windows])
        max_ax = max([window.ax_max for window in self.windows])
        min_lat = min([window.lat_min for window in self.windows])
        max_lat = max([window.lat_max for window in self.windows])
        if self.image_data.bmode.ndim == 3:
            min_cor = min([window.cor_min for window in self.windows])
            max_cor = max([window.cor_max for window in self.windows])
        
        self.single_window = Window()
        self.single_window.lat_min = min_lat
        self.single_window.lat_max = max_lat
        self.single_window.ax_min = min_ax
        self.single_window.ax_max = max_ax
        if self.image_data.bmode.ndim == 3:
            self.single_window.cor_min = min_cor
            self.single_window.cor_max = max_cor
        
        # Get the full signal data
        if self.image_data.bmode.ndim == 2:
            img_window = self.image_data.rf_data[
                self.single_window.ax_min: self.single_window.ax_max + 1,
                self.single_window.lat_min: self.single_window.lat_max + 1
            ]
            phantom_window = self.image_data.phantom_rf_data[
                self.single_window.ax_min: self.single_window.ax_max + 1,
                self.single_window.lat_min: self.single_window.lat_max + 1
            ]
        elif self.image_data.bmode.ndim == 3:
            img_window = self.image_data.rf_data[
                self.single_window.cor_min: self.single_window.cor_max + 1,
                self.single_window.lat_min: self.single_window.lat_max + 1,
                self.single_window.ax_min: self.single_window.ax_max + 1
            ]
            phantom_window = self.image_data.phantom_rf_data[
                self.single_window.cor_min: self.single_window.cor_max + 1,
                self.single_window.lat_min: self.single_window.lat_max + 1,
                self.single_window.ax_min: self.single_window.ax_max + 1
            ]
        
        # Process full signal functions
        for i, function in enumerate(self.ordered_funcs):
            if self.ordered_func_names[i] in self.unordered_full_seg_func_names:
                function(img_window, phantom_window, self.single_window, self.config, self.image_data, **self.analysis_kwargs)
