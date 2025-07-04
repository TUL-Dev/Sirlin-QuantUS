import json
import yaml
import argparse
from pathlib import Path

from quantus.image_loading.utc_loaders.options import get_scan_loaders, scan_loader_args
from quantus.seg_loading.options import get_seg_loaders, seg_loader_args
from quantus.analysis_config.utc_config.options import get_config_loaders, config_loader_args
from quantus.analysis.options import get_analysis_types, analysis_args
from quantus.visualizations.options import get_visualization_types, visualization_args
from quantus.data_export.options import get_data_export_types, data_export_args

DESCRIPTION = """
QuantUS | Custom US Analysis Workflows
"""
    
def main_cli() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    scan_loader_args(parser)
    seg_loader_args(parser)
    config_loader_args(parser)
    analysis_args(parser)
    visualization_args(parser)
    data_export_args(parser)
    args = parser.parse_args()
    args.scan_loader_kwargs = json.loads(args.scan_loader_kwargs)
    args.seg_loader_kwargs = json.loads(args.seg_loader_kwargs)
    args.config_kwargs = json.loads(args.config_kwargs)
    args.analysis_kwargs = json.loads(args.analysis_kwargs)
    args.visualization_kwargs = json.loads(args.visualization_kwargs)
    
    return core_pipeline(args)    

def main_yaml() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('config', type=str, help='Path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        args = argparse.Namespace(**config, **vars(args))
    args.scan_loader_kwargs = {} if args.scan_loader_kwargs is None else args.scan_loader_kwargs
    args.seg_loader_kwargs = {} if args.seg_loader_kwargs is None else args.seg_loader_kwargs
    args.config_kwargs = {} if args.config_kwargs is None else args.config_kwargs
    args.analysis_kwargs = {} if args.analysis_kwargs is None else args.analysis_kwargs
    args.visualization_kwargs = {} if args.visualization_kwargs is None else args.visualization_kwargs
    
    return core_pipeline(args)

def main_dict(config: dict) -> int:
    """Runs the full QuantUS workflow from a config dictionary.
    
    Args:
        config (dict): Configuration dictionary with all necessary parameters.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    args = argparse.Namespace(**config)
    args.scan_loader_kwargs = {} if args.scan_loader_kwargs is None else args.scan_loader_kwargs
    args.seg_loader_kwargs = {} if args.seg_loader_kwargs is None else args.seg_loader_kwargs
    args.config_kwargs = {} if args.config_kwargs is None else args.config_kwargs
    args.analysis_kwargs = {} if args.analysis_kwargs is None else args.analysis_kwargs
    args.visualization_kwargs = {} if args.visualization_kwargs is None else args.visualization_kwargs
    
    return core_pipeline(args)
    
def core_pipeline(args) -> int:
    """Runs the full QuantUS workflow. Different from entrypoints in that all requirements are checked at the start rather than dynamically.
    """
    scan_loaders = get_scan_loaders()
    seg_loaders = get_seg_loaders()
    config_loaders = get_config_loaders()
    analysis_types, analysis_funcs = get_analysis_types()
    visualization_types, visualization_funcs = get_visualization_types()
    data_export_types, data_export_funcs = get_data_export_types()
    
    # Get applicable plugins
    try:
        scan_loader = scan_loaders[args.scan_loader]['cls']
        assert Path(args.scan_path).suffix in scan_loaders[args.scan_loader]['file_exts'], f"File must end with {scan_loaders[args.scan_loader]['file_exts']}"
    except KeyError:
        print(f'Parser "{args.scan_loader}" is not available!')
        print(f"Available parsers: {', '.join(scan_loaders.keys())}")
        return 1
    try:
        seg_loader = seg_loaders[args.seg_loader]['func']
    except KeyError:
        print(f'Segmentation loader "{args.seg_loader}" is not available!')
        print(f"Available segmentation loaders: {', '.join(seg_loaders.keys())}")
        return 1
    try:
        config_loader = config_loaders[args.config_loader]['func']
    except KeyError:
        print(f'Analysis config loader "{args.config_loader}" is not available!')
        print(f"Available analysis config loaders: {', '.join(config_loaders.keys())}")
        return 1
    try:
        analysis_class = analysis_types[args.analysis_type]
    except KeyError:
        print(f'Analysis type "{args.analysis_type}" is not available!')
        print(f"Available analysis types: {', '.join(analysis_types.keys())}")
        return 1
    try:
        visualization_class = visualization_types[args.visualization_type]
    except KeyError:
        print(f'Visualization type "{args.visualization_type}" is not available!')
        print(f"Available visualization types: {', '.join(visualization_types.keys())}")
        return 1
    try:
        data_export_class = data_export_types[args.data_export_type]
    except KeyError:
        print(f'Data export type "{args.data_export_type}" is not available!')
        print(f"Available data export types: {', '.join(data_export_types.keys())}")
        return 1
    
    # Check scan paths
    try:
        assertions = [args.scan_path.endswith(ext) for ext in scan_loaders[args.scan_loader]['file_exts']]
        assert max(assertions), f"Scan file must end with {', '.join(scan_loaders[args.scan_loader]['file_exts'])}"
    except KeyError:
        print(f"Scan loader '{args.scan_loader}' does not have defined file extensions.")
        print(f"Available scan loaders: {', '.join(scan_loaders.keys())}")
        return 1

    # Check segmentation paths
    try:
        assertions = [args.seg_path.endswith(ext) for ext in seg_loaders[args.seg_loader]['exts']]
        assert not len(assertions) or max(assertions), f"Segmentation file must end with {', '.join(seg_loaders[args.seg_loader]['exts'])}"
    except KeyError:
        print(f"Segmentation loader '{args.seg_loader}' does not have defined file extensions.")
        print(f"Available segmentation loaders: {', '.join(seg_loaders.keys())}")
        return 1

    # Check config paths
    try:
        assertions = [args.config_path.endswith(ext) for ext in config_loaders[args.config_loader]['exts']]
        assert not len(assertions) or max(assertions), f"Config file must end with {', '.join(config_loaders[args.config_loader]['exts'])}"
    except KeyError:
        print(f"Config loader '{args.config_loader}' does not have defined file extensions.")
        print(f"Available config loaders: {', '.join(config_loaders.keys())}")
        return 1

    # Check analysis setup
    if args.analysis_funcs is None:
        args.analysis_funcs = list(analysis_funcs[args.analysis_type].keys())
    for name in args.analysis_funcs:
        assert analysis_funcs[args.analysis_type].get(name) is not None, f"Function '{name}' not found in {args.analysis_type} analysis type.\nAvailable functions: {', '.join(analysis_funcs[args.analysis_type].keys())}"
        analysis_kwargs = analysis_funcs[args.analysis_type][name].get('kwarg_names', [])
        for kwarg in analysis_kwargs:
            if kwarg not in args.analysis_kwargs:
                raise ValueError(f"analysis_kwargs: Missing required keyword argument '{kwarg}' for function '{name}' in {args.analysis_type} analysis type.")
            
    # Check visualization setup
    if args.visualization_funcs is None:
        args.visualization_funcs = []
    for name in args.visualization_funcs:
        if name == "paramaps":
            continue
        assert visualization_funcs[args.visualization_type].get(name) is not None, f"Function '{name}' not found in {args.visualization_type} visualization type.\nAvailable functions: {', '.join(visualization_funcs[args.visualization_type].keys())}"
        
    # Check data export setup
    if args.data_export_funcs is None:
        args.data_export_funcs = []
    for name in args.data_export_funcs:
        assert data_export_funcs[args.data_export_type].get(name) is not None, f"Function '{name}' not found in {args.data_export_type} data export type.\nAvailable functions: {', '.join(data_export_funcs[args.data_export_type].keys())}"
    
    # Parsing / data loading
    image_data = scan_loader(args.scan_path, args.phantom_path, **args.scan_loader_kwargs) # Load signal data
    seg_data = seg_loader(image_data, args.seg_path, scan_path=args.scan_path, phantom_path=args.phantom_path, **args.seg_loader_kwargs) # Load seg data
    config = config_loader(args.config_path, scan_path=args.scan_path, phantom_path=args.phantom_path, **args.config_kwargs) # Load analysis config
    
    # Analysis
    if image_data.spatial_dims < image_data.rf_data.ndim:
        image_data.rf_data = image_data.rf_data[seg_data.frame]
        image_data.bmode = image_data.bmode[seg_data.frame]
        if image_data.sc_bmode is not None:
            image_data.sc_bmode = image_data.sc_bmode[seg_data.frame]
        assert image_data.bmode.ndim == image_data.spatial_dims, \
            "Bmode data dimensions do not match spatial dimensions!"
    elif image_data.spatial_dims > image_data.rf_data.ndim:
        raise ValueError("Spatial dimensions are greater than RF data dimensions!")
    analysis_obj = analysis_class(image_data, config, seg_data, args.analysis_funcs, **args.analysis_kwargs)
    analysis_obj.compute_paramaps()
    analysis_obj.compute_single_window()

    # Visualizations
    visualization_obj = visualization_class(analysis_obj, args.visualization_funcs, **args.visualization_kwargs)
    visualization_obj.export_visualizations()
    
    # Numerical data export
    data_export_obj = data_export_class(visualization_obj, args.data_export_path, args.data_export_funcs)
    data_export_obj.save_data()
    
    return 0

if __name__ == '__main__':
    exit(main_yaml())
