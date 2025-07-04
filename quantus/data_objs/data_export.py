from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .visualizations import ParamapDrawingBase
        

class BaseDataExport(ABC):
    """Export numerical data from parametric map analysis.
    """
    def __init__(self, visualizations_obj: ParamapDrawingBase, output_path: str):
        self.visualizations_obj = visualizations_obj
        self.export_path = output_path
        self.exported_df: pd.DataFrame = None
        
    @abstractmethod
    def save_data(self):
        """Construct pandas dataframe to export results and save."""
        data_dict = {}
        data_dict["Scan Name"] = self.visualizations_obj.analysis_obj.image_data.scan_name
        data_dict["Phantom Name"] = self.visualizations_obj.analysis_obj.image_data.phantom_name
        data_dict["Seg Name"] = self.visualizations_obj.analysis_obj.seg_data.seg_name
        
        return data_dict
