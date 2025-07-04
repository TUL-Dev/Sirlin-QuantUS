import pandas as pd
from typing import List

from ...data_objs.visualizations import ParamapDrawingBase
from ...data_objs.data_export import BaseDataExport
from .functions import *

class CSVExport(BaseDataExport):
    """Export parametric map data to CSV format.
    """
    def __init__(self, visualizations_obj: ParamapDrawingBase, output_path: str, function_names: List[str]):
        super().__init__(visualizations_obj, output_path)
        self.function_names = function_names
        assert output_path.endswith(".csv"), "Output path must end with .csv to export to CSV format."
        
    def save_data(self):
        """Data saved dynamically to a dict object and eventually converted to a pandas dataframe and saved to a CSV file.
        It is recommended to only save numerical or string data types to avoid issues with CSV format. For more 
        complex data types (e.g. lists, dicts), consider using the PKLExport option instead.
        """
        data_dict = super().save_data()
        
        for function_name in self.function_names:
            function = globals()[function_name]
            function(self.visualizations_obj, data_dict)
            
        if len(self.function_names):
            self.exported_df = pd.DataFrame(data_dict)
            self.exported_df.to_csv(self.export_path, index=False)
        else:
            print("No CSV data exported. No export functions provided.")
        