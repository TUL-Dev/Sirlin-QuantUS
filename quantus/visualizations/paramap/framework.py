import pickle
from pathlib import Path
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ...data_objs.visualizations import ParamapDrawingBase
from ...data_objs.analysis import ParamapAnalysisBase
from ...data_objs.image import UltrasoundRfImage
from ...data_objs.seg import BmodeSeg
from .functions import *

class_name = "ParamapAnalysis"

class ParamapVisualizations(ParamapDrawingBase):
    """
    Class to complete visualizations of parametric map-based UTC analysis.
    """

    def __init__(self, analysis_obj: ParamapAnalysisBase, visualization_funcs: list, **kwargs):
        # Type checking
        assert isinstance(analysis_obj, ParamapAnalysisBase), "analysis_obj must be a ParamapAnalysisBase child class"
        super().__init__(analysis_obj)
        
        self.paramap_folder_path = kwargs.get('paramap_folder_path', "Visualizations")
        
        self.analysis_obj = analysis_obj
        self.visualization_funcs = visualization_funcs
        self.kwargs = kwargs
        self.paramaps = []
        self.legend_paramaps = []
        self.plots = []
        
    def save_2d_paramap(self, bmode: np.ndarray, paramap: np.ndarray, legend: plt.Figure, dest_path: Path) -> None:
        """Saves the parametric map and legend to the specified path.
        
        Args:
            bmode (np.ndarray): The B-mode image to save.
            paramap (np.ndarray): The parametric map to save.
            legend (plt.Figure): The legend figure to save.
            dest_path (str): The destination path for saving the parametric map.
        """
        assert str(dest_path).endswith('.png'), "Parametric map output path must end with .png"
        
        # Overlay the paramap on the B-mode image
        fig, ax = plt.subplots()
        if self.analysis_obj.image_data.sc_bmode is not None:
            width = paramap.shape[1]*self.analysis_obj.image_data.sc_lateral_res
            height = paramap.shape[0]*self.analysis_obj.image_data.sc_axial_res
        else:
            width = paramap.shape[1]*self.analysis_obj.image_data.lateral_res
            height = paramap.shape[0]*self.analysis_obj.image_data.axial_res
        aspect = width/height
        im = ax.imshow(bmode)
        im = ax.imshow(paramap)
        extent = im.get_extent()
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
        ax.axis('off')
        
        fig.savefig(dest_path.parent / (dest_path.stem + '_paramap.png'), bbox_inches='tight', pad_inches=0)
        legend.savefig(dest_path.parent / (dest_path.stem + '_legend.png'), bbox_inches='tight', pad_inches=0)
        
    def save_general_paramap(self, paramap: np.ndarray, legend: plt.Figure, dest_path: Path) -> None:
        """Saves the parametric map to the specified path.
        
        Args:
            paramap (np.ndarray): The parametric map to save.
            dest_path (str): The destination path for saving the parametric map.
        """
        assert str(dest_path).endswith('.pkl'), "Parametric map output path must end with .pkl"
        
        legend.savefig(dest_path.parent / (dest_path.stem + '_legend.png'), bbox_inches='tight', pad_inches=0)
        
        with open(dest_path.parent / (dest_path.stem + '_paramap.pkl'), 'wb') as f:
            pickle.dump(paramap, f)
        
    def export_visualizations(self):
        """Used to specify which visualizations to export and where.
        """
        if len(self.visualization_funcs):
            paramap_folder_path = Path(self.paramap_folder_path)
            paramap_folder_path.mkdir(parents=True, exist_ok=True)
        
        if "paramaps" in self.visualization_funcs:
            if len(self.analysis_obj.image_data.bmode.shape) == 2:
                if self.analysis_obj.image_data.sc_bmode is not None:
                    bmode = cv2.cvtColor(np.array(self.analysis_obj.image_data.sc_bmode, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
                else:
                    bmode = cv2.cvtColor(np.array(self.analysis_obj.image_data.bmode, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
            else:
                bmode = None
                if self.analysis_obj.image_data.sc_bmode is not None:
                    with open(paramap_folder_path / 'bmode.pkl', 'wb') as f:
                        pickle.dump(self.analysis_obj.image_data.sc_bmode, f)
                    with open(paramap_folder_path / 'segmentation.pkl', 'wb') as f:
                        pickle.dump(self.analysis_obj.seg_data.sc_seg_mask, f)
                else:
                    with open(paramap_folder_path / 'bmode.pkl', 'wb') as f:
                        pickle.dump(self.analysis_obj.image_data.bmode, f)
                    with open(paramap_folder_path / 'segmentation.pkl', 'wb') as f:
                        pickle.dump(self.analysis_obj.seg_data.seg_mask, f)
            
            # Save parametric maps
            params = self.analysis_obj.windows[0].results.__dict__.keys()
            cmap_ix = 0
            for param in params:
                if isinstance(getattr(self.analysis_obj.windows[0].results, param), (str, list, np.ndarray)):
                    continue
                paramap, legend = self.draw_paramap(param, self.cmaps[cmap_ix % len(self.cmaps)])
                cmap_ix += 1
                
                if bmode is not None:
                    self.save_2d_paramap(bmode, paramap, legend, paramap_folder_path / f'{param}.png')
                else:
                    self.save_general_paramap(paramap, legend, paramap_folder_path / f'{param}.pkl')

        # Complete all custom visualizations
        for func_name in self.visualization_funcs:
            if func_name == "paramaps":
                continue
            function = globals()[func_name]
            function(self.analysis_obj, self.paramap_folder_path)
