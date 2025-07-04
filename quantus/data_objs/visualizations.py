from typing import Tuple
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .analysis import ParamapAnalysisBase
from ..image_loading.utc_loaders.transforms import scanConvert, scanConvert3dVolumeSeries

class ParamapDrawingBase(ABC):
    """Facilitate parametric map visualizations of ultrasound images.
    """
    
    def __init__(self, analysis_obj: ParamapAnalysisBase, **kwargs):
        # Cmap library
        summer_cmap = plt.get_cmap("summer")
        summer_cmap = [summer_cmap(i)[:3] for i in range(summer_cmap.N)]
        winter_cmap = plt.get_cmap("winter")
        winter_cmap = [winter_cmap(i)[:3] for i in range(winter_cmap.N)]
        autumn_cmap = plt.get_cmap("autumn")  # Fixed typo in "autunm"
        autumn_cmap = [autumn_cmap(i)[:3] for i in range(autumn_cmap.N)]
        spring_cmap = plt.get_cmap("spring")
        spring_cmap = [spring_cmap(i)[:3] for i in range(spring_cmap.N)]
        cool_cmap = plt.get_cmap("cool")
        cool_cmap = [cool_cmap(i)[:3] for i in range(cool_cmap.N)]
        hot_cmap = plt.get_cmap("hot")
        hot_cmap = [hot_cmap(i)[:3] for i in range(hot_cmap.N)]
        bone_cmap = plt.get_cmap("bone")
        bone_cmap = [bone_cmap(i)[:3] for i in range(bone_cmap.N)]
        copper_cmap = plt.get_cmap("copper")
        copper_cmap = [copper_cmap(i)[:3] for i in range(copper_cmap.N)]
        jet_cmap = plt.get_cmap("jet")
        jet_cmap = [jet_cmap(i)[:3] for i in range(jet_cmap.N)]
        self.cmaps = [np.array(plt.get_cmap("viridis").colors), np.array(plt.get_cmap("magma").colors),
                    np.array(plt.get_cmap("plasma").colors), np.array(plt.get_cmap("inferno").colors),
                    np.array(plt.get_cmap("cividis").colors), np.array(summer_cmap),
                    np.array(winter_cmap), np.array(autumn_cmap), np.array(spring_cmap),
                    np.array(cool_cmap), np.array(hot_cmap), np.array(bone_cmap), np.array(copper_cmap),
                    np.array(jet_cmap)]
        self.cmap_names = ["viridis", "magma", "plasma", "inferno", "cividis", "summer", "winter", "autumn",
                        "spring", "cool", "hot", "bone", "copper", "jet"]
        
        self.analysis_obj = analysis_obj
        self.paramaps = []
        self.legend_paramaps = []
        
        assert len(self.analysis_obj.windows), "No analyzed windows to visualize"
        self.window_idx_map = np.zeros_like(self.analysis_obj.image_data.bmode, dtype=int)
        for i, window in enumerate(self.analysis_obj.windows):
            if window.cor_min == -1:
                self.window_idx_map[window.ax_min: window.ax_max+1, window.lat_min: window.lat_max+1
                                    ] = i+1
            else:
                self.window_idx_map[window.cor_min: window.cor_max+1, window.lat_min: window.lat_max+1, window.ax_min: window.ax_max+1
                                    ] = i+1
                
        if self.analysis_obj.image_data.sc_bmode is not None:
            image_data = self.analysis_obj.image_data
            if image_data.sc_bmode.ndim == 2:
                self.sc_window_idx_map = np.array(scanConvert(self.window_idx_map, image_data.width,
                                                    image_data.tilt, image_data.start_depth,
                                                    image_data.end_depth, desiredHeight=image_data.sc_bmode.shape[0])[0].scArr,
                                                  dtype=int)
            elif image_data.sc_bmode.ndim == 3:
                self.sc_window_idx_map = np.array(scanConvert3dVolumeSeries(self.window_idx_map, image_data.sc_params_3d, scale=False, interp='nearest')[0],
                                                  dtype=int)
            else:
                raise NotImplementedError("Scan conversion only implemented for 2D and 3D data.")
            
        
    def draw_paramap(self, param: str, cmap: list) -> Tuple[np.ndarray, plt.Figure]:
        """Draws RGB parametric map for parameter with a specified colormap and get the legend.
        """
        assert hasattr(self.analysis_obj.windows[0].results, param), "Given parameter cannot be found in results"
        
        param_vals = []
        for window in self.analysis_obj.windows:
            param_val = getattr(window.results, param)
            param_vals.append(param_val)
        min_val = min(param_vals); max_val = max(param_vals)
         
        idx_map = self.window_idx_map if self.analysis_obj.image_data.sc_bmode is None else self.sc_window_idx_map
        window_points = np.transpose(np.where(idx_map > 0))
        colored_paramap = np.zeros(idx_map.shape + (4,), dtype=np.uint8)
        for point in window_points:
            window = self.analysis_obj.windows[int(np.round(idx_map[*point])-1)]
            color_ix = int((255 / (max_val-min_val)
                                     )*(getattr(window.results, param)-min_val)
                                    ) if min_val != max_val else 125
            colored_paramap[*point, :3] = (np.array(cmap[color_ix])*255).astype(np.uint8)
            colored_paramap[*point, 3] = 255
            
        # Plot the paramap legend
        fig, ax = plt.subplots(figsize=(2, 10))
        gradient = np.linspace(0, 1, len(cmap)).reshape(-1, 1)
        ax.imshow(gradient, aspect='auto', cmap=mpl.colors.ListedColormap(cmap))
        ax.set_yticks(np.linspace(0, len(cmap), 6))
        ax.set_yticklabels([f"{np.round(min_val + i*((max_val-min_val)/5), 2)}"
                            for i in np.linspace(0, 5, 6)])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.invert_yaxis()
        
        return colored_paramap, fig

    @abstractmethod
    def export_visualizations(self):
        """Used to specify which visualizations to export and where.
        """
        pass
