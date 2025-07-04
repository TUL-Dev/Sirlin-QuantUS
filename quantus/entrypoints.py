import copy

from quantus.data_objs import UltrasoundRfImage, BmodeSeg, RfAnalysisConfig, \
        ParamapAnalysisBase, ParamapDrawingBase, BaseDataExport
from quantus.image_loading.utc_loaders.options import get_scan_loaders
from quantus.seg_loading.options import get_seg_loaders
from quantus.analysis_config.utc_config.options import get_config_loaders
from quantus.analysis.options import get_analysis_types
from quantus.visualizations.options import get_visualization_types
from quantus.data_export.options import get_data_export_types

DESCRIPTION = """
QuantUS | Custom US Analysis Workflows
"""

def scan_loading_step(scan_type: str, scan_path: str, phantom_path: str, **scan_loader_kwargs) -> UltrasoundRfImage:
    """Load the scan data using the specified scan loader.

    Args:
        scan_type (str): The type of scan loader to use.
        scan_path (str): Path to the scan file.
        phantom_path (str): Path to the phantom file.
        **scan_loader_kwargs: Additional keyword arguments for the scan loader.

    Returns:
        UltrasoundRfImage: Loaded ultrasound RF image data.
    """
    scan_loaders = get_scan_loaders()
    
    # Find the scan loader
    try:
        scan_loader = scan_loaders[scan_type]['cls']
        assertions = [scan_path.endswith(ext) for ext in scan_loaders[scan_type]['file_exts']]
        assert max(assertions), f"Scan file must end with {', '.join(scan_loaders[scan_type]['file_exts'])}"
    except KeyError:
        print(f'Parser "{scan_type}" is not available!')
        print(f"Available parsers: {', '.join(scan_loaders.keys())}")
        return 1
    
    image_data: UltrasoundRfImage = scan_loader(scan_path, phantom_path, **scan_loader_kwargs)
    image_data.spatial_dims = scan_loaders[scan_type]['spatial_dims']
    return image_data

def seg_loading_step(seg_type: str, image_data: UltrasoundRfImage, seg_path: str, 
                     scan_path: str, phantom_path: str, **seg_loader_kwargs) -> BmodeSeg:
    """Load the segmentation data using the specified segmentation loader.

    Args:
        seg_type (str): The type of segmentation loader to use.
        image_data (UltrasoundRfImage): Loaded ultrasound RF image data.
        seg_path (str): Path to the segmentation file.
        scan_path (str): Path to the scan file.
        phantom_path (str): Path to the phantom file.
        **seg_loader_kwargs: Additional keyword arguments for the segmentation loader.

    Returns:
        BmodeSeg: Loaded segmentation data.
    """
    seg_loaders = get_seg_loaders()
    
    # Find the segmentation loader
    try:
        seg_loader = seg_loaders[seg_type]['func']
        assertions = [seg_path.endswith(ext) for ext in seg_loaders[seg_type]['exts']]
        assert not len(assertions) or max(assertions), f"Segmentation file must end with {', '.join(seg_loaders[seg_type]['exts'])}"
    except KeyError:
        print(f'Segmentation loader "{seg_type}" is not available!')
        print(f"Available segmentation loaders: {', '.join(seg_loaders.keys())}")
        return 1
    
    return seg_loader(image_data, seg_path, scan_path=scan_path, phantom_path=phantom_path, **seg_loader_kwargs)

def analysis_config_step(config_type: str, config_path: str, scan_path: str, phantom_path: str, **config_kwargs) -> RfAnalysisConfig:
    """Load the analysis configuration using the specified config loader.

    Args:
        config_type (str): The type of config loader to use.
        config_path (str): Path to the config file.
        scan_path (str): Path to the scan file.
        phantom_path (str): Path to the phantom file.
        **config_kwargs: Additional keyword arguments for the config loader.

    Returns:
        RfAnalysisConfig: Loaded analysis configuration.
    """
    config_loaders = get_config_loaders()
    
    # Find the config loader
    try:
        config_loader = config_loaders[config_type]['func']
        assertions = [config_path.endswith(ext) for ext in config_loaders[config_type]['exts']]
        assert not len(assertions) or max(assertions), f"Config file must end with {', '.join(config_loaders[config_type]['exts'])}"
    except KeyError:
        print(f'Analysis config loader "{config_type}" is not available!')
        print(f"Available analysis config loaders: {', '.join(config_loaders.keys())}")
        return 1
    
    return config_loader(config_path, scan_path=scan_path, phantom_path=phantom_path, **config_kwargs)

def analysis_step(analysis_type: str, image_data: UltrasoundRfImage, config: RfAnalysisConfig,
                  seg_data: BmodeSeg, analysis_funcs: list, **analysis_kwargs) -> ParamapAnalysisBase:
    """Perform analysis using the specified analysis type.
    
    Args:
        analysis_type (str): The type of analysis to perform.
        image_data (UltrasoundRfImage): Loaded ultrasound RF image data.
        config (RfAnalysisConfig): Loaded analysis configuration.
        seg_data (BmodeSeg): Loaded segmentation data.
        analysis_funcs (list): List of analysis functions to apply.
        **analysis_kwargs: Additional keyword arguments for the analysis.
    Returns:
        ParamapAnalysisBase: Analysis object containing the results.
    """
    all_analysis_types, all_analysis_funcs = get_analysis_types()
    
    # Find the analysis class
    try:
        analysis_class = all_analysis_types[analysis_type]
    except KeyError:
        print(f'Analysis type "{analysis_type}" is not available!')
        print(f"Available analysis types: {', '.join(all_analysis_types.keys())}")
        return 1
    
    # Check analysis setup
    for name in analysis_funcs:   
        if name not in all_analysis_funcs[analysis_type]:
            raise ValueError(f"Function '{name}' not found in {analysis_type} analysis type.\nAvailable functions: {', '.join(all_analysis_funcs[analysis_type].keys())}")
        required_analysis_kwargs = all_analysis_funcs[analysis_type][name].get('kwarg_names', [])
        for kwarg in required_analysis_kwargs:
            if kwarg not in analysis_kwargs:
                raise ValueError(f"analysis_kwargs: Missing required keyword argument '{kwarg}' for function '{name}' in {analysis_type} analysis type.")
            
    # Perform analysis
    analyzed_image_data = copy.deepcopy(image_data)
    if analyzed_image_data.spatial_dims < analyzed_image_data.rf_data.ndim:
        analyzed_image_data.rf_data = analyzed_image_data.rf_data[seg_data.frame]
        analyzed_image_data.bmode = analyzed_image_data.bmode[seg_data.frame]
        if analyzed_image_data.sc_bmode is not None:
            analyzed_image_data.sc_bmode = analyzed_image_data.sc_bmode[seg_data.frame]
        assert analyzed_image_data.bmode.ndim == analyzed_image_data.spatial_dims, \
            "Bmode data dimensions do not match spatial dimensions!"
    elif analyzed_image_data.spatial_dims > analyzed_image_data.rf_data.ndim:
        raise ValueError("Spatial dimensions are greater than RF data dimensions!")
    
    analysis_obj = analysis_class(analyzed_image_data, config, seg_data, analysis_funcs, **analysis_kwargs)
    analysis_obj.compute_paramaps()
    analysis_obj.compute_single_window()
    
    return analysis_obj

def visualization_step(visualization_type: str, analysis_obj: ParamapAnalysisBase,
                       visualization_funcs: list, **visualization_kwargs) -> ParamapDrawingBase:
    """Perform visualization using the specified visualization type.
    
    Args:
        visualization_type (str): The type of visualization to perform.
        analysis_obj (ParamapAnalysisBase): Analysis object containing the results.
        visualization_funcs (list): List of visualization functions to apply.
        **visualization_kwargs: Additional keyword arguments for the visualization.
    Returns:
        ParamapDrawingBase: Visualization object containing the results.
    """
    all_visualization_types, all_visualization_funcs = get_visualization_types()
    
    # Find the visualization class
    try:
        visualization_class = all_visualization_types[visualization_type]
    except KeyError:
        print(f'Visualization type "{visualization_type}" is not available!')
        print(f"Available visualization types: {', '.join(all_visualization_types.keys())}")
        return 1
    
    # Check visualization setup
    for name in visualization_funcs:
        if name not in all_visualization_funcs[visualization_type]:
            raise ValueError(f"Function '{name}' not found in {visualization_type} visualization type.\nAvailable functions: {', '.join(all_visualization_funcs[visualization_type].keys())}")
    
    # Perform visualization
    visualization_obj = visualization_class(analysis_obj, visualization_funcs, **visualization_kwargs)
    visualization_obj.export_visualizations()
    
    return visualization_obj

def data_export_step(data_export_type: str, visualization_obj: ParamapDrawingBase,
                     data_export_path: str, data_export_funcs: list) -> BaseDataExport:
    """Export data using the specified data export type.
    
    Args:
        data_export_type (str): The type of data export to perform.
        visualization_obj (ParamapDrawingBase): Visualization object containing the results.
        data_export_path (str): Path to save the exported data.
        data_export_funcs (list): List of data export functions to apply.
    Returns:
        BaseDataExport: Data export object containing the results.
    """
    all_data_export_types, all_data_export_funcs = get_data_export_types()
    
    # Find the data export class
    try:
        data_export_class = all_data_export_types[data_export_type]
    except KeyError:
        print(f'Data export type "{data_export_type}" is not available!')
        print(f"Available data export types: {', '.join(all_data_export_types.keys())}")
        return 1
    
    # Check data export setup
    for name in data_export_funcs:
        if name not in all_data_export_funcs[data_export_type].keys():
            raise ValueError(f"Function '{name}' not found in {data_export_type} data export type.\nAvailable functions: {', '.join(data_export_funcs[data_export_type].keys())}")
    
    # Perform data export
    data_export_obj = data_export_class(visualization_obj, data_export_path, data_export_funcs)
    data_export_obj.save_data()
    
    return data_export_obj
