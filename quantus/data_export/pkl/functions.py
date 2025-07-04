import numpy as np
from typing import Dict

from ...data_objs.visualizations import ParamapDrawingBase
        
def paramaps(visualizations_obj: ParamapDrawingBase, data_dict: Dict[str, str]) -> None:
    """Save all parametric map arrays. Each window will have its corresponding numerical value
    instead of a RGB color.

    Args:
        visualizations_obj (ParamapDrawingBase): The visualizations object containing the data.
        params (Dict[str, str]): Dictionary of parameters to compute mean values for.
    """
    params = visualizations_obj.analysis_obj.windows[0].results.__dict__.keys()
    if hasattr(visualizations_obj, "sc_window_idx_map"):
        idx_map = visualizations_obj.sc_window_idx_map
        data_dict["bmode"] = [visualizations_obj.analysis_obj.image_data.sc_bmode]
    else:
        idx_map = visualizations_obj.window_idx_map
        data_dict["bmode"] = [visualizations_obj.analysis_obj.image_data.bmode]
        
    data_dict["mask"] = [idx_map > 0]
    
    for param in params:
        if isinstance(getattr(visualizations_obj.analysis_obj.windows[0].results, param), (str, list, np.ndarray)):
            continue
        paramap = np.zeros_like(idx_map, dtype=float)
        for i, window in enumerate(visualizations_obj.analysis_obj.windows):
            paramap[idx_map == i+1] = getattr(window.results, param)
        data_dict[f"{param}_paramap"] = [paramap]
