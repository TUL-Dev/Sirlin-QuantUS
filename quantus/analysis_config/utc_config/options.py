import importlib
from pathlib import Path

from argparse import ArgumentParser

from .functions import *

def config_loader_args(parser: ArgumentParser):
    parser.add_argument('config_path', type=str, help='Path to analysis config')
    parser.add_argument('--config_type', type=str, default='pkl_utc',
                        help='Analysis config loader to use. See "quantus_parse.analysis_config_loaders" in pyproject.toml for available analysis config loaders.')
    parser.add_argument('--config_kwargs', type=str, default='{}',
                        help='Analysis config kwargs in JSON format needed for analysis class.')
    
    
def get_config_loaders() -> dict:
    """Get scan loaders for the CLI.
    
    Returns:
        dict: Dictionary of scan loaders.
    """
    functions = {}
    for name, obj in globals().items():
        if type(obj) is dict:
            try:
                if callable(obj['func']) and obj['func'].__module__ == 'quantus.analysis_config.utc_config.functions':
                    functions[name] = {}
                    functions[name]['func'] = obj['func']
                    functions[name]['exts'] = obj['exts']
            except KeyError:
                pass
            
    return functions
