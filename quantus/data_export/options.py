import importlib
from pathlib import Path
from typing import Tuple

from argparse import ArgumentParser

def data_export_args(parser: ArgumentParser):
    parser.add_argument('data_export_type', type=str, default='',
                        help='Data export type to use. Available data export types: ' + ', '.join(get_data_export_types()[0].keys()))
    parser.add_argument('data_export_path', type=str,
                        help='Path to save exported numerical data to. Must end in .csv or .pkl')
    parser.add_argument('--data_export_kwargs', type=str, default='{}',
                        help='Data export kwargs in JSON format needed for data export class.')
    

def get_data_export_types() -> Tuple[dict, dict]:
    """Get visualization types for the CLI.
    
    Returns:
        dict: Dictionary of visualization types.
        dict: Dictionary of visualization functions for each type.
    """
    types = {}
    current_dir = Path(__file__).parent
    for folder in current_dir.iterdir():
        # Check if the item is a directory and not a hidden directory
        if folder.is_dir() and not folder.name.startswith("_"):
            try:
                # Attempt to import the module
                module = importlib.import_module(
                    f"quantus.data_export.{folder.name}.framework"
                )
                entry_class = getattr(module, f"{folder.name.upper()}Export", None)
                if entry_class:
                    types[folder.name] = entry_class
            except ModuleNotFoundError:
                # Handle the case where the module cannot be found
                pass
            
    functions = {}
    for type_name, _ in types.items():
        try:
            module = importlib.import_module(f'quantus.data_export.{type_name}.functions')
            for name, obj in vars(module).items():
                try:
                    if callable(obj) and obj.__module__ == f'quantus.data_export.{type_name}.functions':
                        if not isinstance(obj, type):
                            functions[type_name] = functions.get(type_name, {})
                            functions[type_name][name] = obj
                except (TypeError, KeyError):
                    pass
        except ModuleNotFoundError:
            # Handle the case where the functions module cannot be found
            functions[type_name] = {}
            
    return types, functions
