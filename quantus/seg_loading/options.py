from pathlib import Path

from argparse import ArgumentParser

from .functions import *

def seg_loader_args(parser: ArgumentParser):
    parser.add_argument('seg_path', type=str, help='Path to segmentation file')
    parser.add_argument('--seg_type', type=str, default='pkl_roi',
                        help='Segmentation loader to use. Available options: ' + ', '.join(get_seg_loaders().keys()))
    parser.add_argument('--seg_loader_kwargs', type=str, default='{}',
                        help='Segmentation kwargs in JSON format needed for analysis class.')
    
def get_seg_loaders() -> dict:
    """Get scan loaders for the CLI.
    
    Returns:
        dict: Dictionary of scan loaders.
    """
    functions = {}
    for name, obj in globals().items():
        if type(obj) is dict:
            try:
                if callable(obj['func']) and obj['func'].__module__ == 'quantus.seg_loading.functions':
                    functions[name] = {}
                    functions[name]['func'] = obj['func']
                    functions[name]['exts'] = obj['exts']
            except KeyError:
                pass
            
    return functions
