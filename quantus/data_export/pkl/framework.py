import pandas as pd
from typing import List

from ...data_objs.visualizations import ParamapDrawingBase
from ...data_objs.data_export import BaseDataExport
from .functions import *

class PKLExport(BaseDataExport):
    """Export parametric map data to PKL format.
    """
    def __init__(self, visualizations_obj: ParamapDrawingBase, output_path: str, function_names: List[str]):
        super().__init__(visualizations_obj, output_path)
        self.function_names = function_names
        assert output_path.endswith(".pkl"), "Output path must end with .pkl to export to PKL format."
        
    def save_data(self):
        """Data saved dynamically to a dict object and eventually converted to a pandas dataframe and saved to a PKL file.
        This approach is more robust for saving compex data types (e.g. lists, dict, etc). If only string and numerical values
        are needed to be saved, it may make more sense to use CSVExport instead for better accessibility outside of Python.
        """
        data_dict = super().save_data()
        
        for function_name in self.function_names:
            function = globals()[function_name]
            function(self.visualizations_obj, data_dict)
            
        if len(self.function_names):
            self.exported_df = pd.DataFrame(data_dict)
            self.exported_df.to_pickle(self.export_path)
        else:
            print("No CSV data exported. No export functions provided.")
        