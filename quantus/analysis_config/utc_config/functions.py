import pickle
from pathlib import Path

from .decorators import extensions
from ...data_objs.analysis_config import RfAnalysisConfig

@extensions(".pkl", ".pickle")
def pkl_rf(analysis_path: str, **kwargs) -> RfAnalysisConfig:
    """Function to load RF analysis configuration data from a pickle file saved from the QuantUS UI.
    
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the analysis config file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the analysis config file name.
    """
    with open(analysis_path, "rb") as f:
        config_pkl: dict = pickle.load(f)
            
    if kwargs.get("assert_scan"):
        assert config_pkl["Image Name"] == Path(kwargs["scan_path"]).name, 'Scan file name mismatch'
    if kwargs.get("assert_phantom"):
        assert config_pkl["Phantom Name"] == Path(kwargs["phantom_path"]).name, 'Phantom file name mismatch'
    
    out = config_pkl["Config"]
    
    return out

@extensions()
def philips_3d_config(analysis_path: str, **kwargs) -> RfAnalysisConfig:
    """Class to load RF analysis configuration data from a pickle file saved from the QuantUS UI.
    
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the analysis config file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the analysis config file name.
    """
    out = RfAnalysisConfig()

    # Frequency parameters
    out.transducer_freq_band = [0, 7000000] # [min, max] (Hz)
    out.analysis_freq_band = [2500000, 5000000] # [lower, upper] (Hz)
    out.center_frequency = 4500000 # Hz
    out.sampling_frequency = 4*out.center_frequency # Hz

    # Windowing parameters
    out.ax_win_size = 10 # axial length per window (mm)
    out.lat_win_size = 10 # lateral length per window (mm)
    out.window_thresh = 0.95 # % of window area required to be considered in ROI
    out.axial_overlap = 0.5 # % of window overlap in axial direction
    out.lateral_overlap = 0.5 # % of window overlap in lateral direction
    
    # 3D scan parameters
    out.cor_win_size = 20 # coronal length per window (mm)
    out.coronal_overlap = 0.5 # % of window overlap in coronal direction

    return out

@extensions()
def clarius_L15_config(analysis_path: str, **kwargs) -> RfAnalysisConfig:
    """Function to load RF analysis configuration data for Clarius ultrasound data.
    
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the analysis config file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the analysis config file name.
    """
    out = RfAnalysisConfig()

    # Frequency parameters
    out.transducer_freq_band = [0, 15e6]  # [min, max] (Hz), up to 15MHz for high-frequency probes
    out.analysis_freq_band = [2e6, 8e6]  # [lower, upper] (Hz), typical analysis range
    out.center_frequency = 5e6  # Hz, typical center frequency for Clarius probes
    out.sampling_frequency = 30e6 # Hz, following Nyquist criterion

    # Windowing parameters - adjusted for smaller windows due to high frequency
    out.ax_win_size = 10  # axial length per window (mm) - reduced from 10mm
    out.lat_win_size = 5  # lateral length per window (mm) - reduced from 10mm
    out.window_thresh = 0.5  # % of window area required to be considered in ROI - reduced from 0.95
    out.axial_overlap = 0.75  # % of window overlap in axial direction - increased from 0.5
    out.lateral_overlap = 0.75  # % of window overlap in lateral direction - increased from 0.5

    # 3D scan parameters
    out.cor_win_size = None  # coronal length per window (mm)
    out.coronal_overlap = None  # % of window overlap in coronal direction

    return out

@extensions()
def clarius_C3_config(analysis_path: str, **kwargs) -> RfAnalysisConfig:
    """Function to load RF analysis configuration data for Clarius ultrasound data.
    
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the analysis config file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the analysis config file name.
    """
    out = RfAnalysisConfig()
    
    # Frequency parameters
    out.transducer_freq_band = [0, 7.5e6]  # [min, max] (Hz), up to 15MHz for high-frequency probes
    out.analysis_freq_band = [1e6, 4e6]  # [lower, upper] (Hz), typical analysis range
    out.center_frequency = 2.5e6  # Hz, typical center frequency for Clarius probes
    out.sampling_frequency = 15e6  # Hz, following Nyquist criterion
    
    # Windowing parameters
    out.ax_win_size = 5  # axial length per window (mm)
    out.lat_win_size = 2.5  # lateral length per window (mm)
    out.window_thresh = 0.95  # % of window area required to be considered in ROI
    out.axial_overlap = 0.5  # % of window overlap in axial direction
    out.lateral_overlap = 0.5  # % of window overlap in lateral direction
    
    out.cor_win_size = None  # coronal length per window (mm)
    out.coronal_overlap = None  # % of window overlap in coronal direction

    return out

@extensions()
def custom(analysis_path: str, **kwargs) -> RfAnalysisConfig:
    """Function to load RF analysis configuration data from a Python dict.
    
    Kwargs:
        transducer_freq_band (list): [min, max] frequency band of the transducer (Hz).
        analysis_freq_band (list): [lower, upper] frequency band for analysis (Hz).
        center_frequency (float): Center frequency of the transducer (Hz).
        sampling_frequency (float): Sampling frequency (Hz).
        ax_win_size (float): Axial length per window (mm).
        lat_win_size (float): Lateral length per window (mm).
        window_thresh (float): Percentage of window area required to be considered in ROI.
        axial_overlap (float): Percentage of window overlap in axial direction.
        lateral_overlap (float): Percentage of window overlap in lateral direction.

        OPTIONAL FOR 3D SCANS:
        cor_win_size (float): Coronal length per window (mm).
        coronal_overlap (float): Percentage of window overlap in coronal direction.
    """
    out = RfAnalysisConfig()
    
    try:
        assert type(kwargs["analysis_freq_band"]) is list, "analysis_freq_band must be a list"
        assert len(kwargs["analysis_freq_band"]) == 2, "analysis_freq_band must be a list of two elements [lower, upper]"
        assert type(kwargs["transducer_freq_band"]) is list, "transducer_freq_band must be a list"
        assert len(kwargs["transducer_freq_band"]) == 2, "transducer_freq_band must be a list of two elements [min, max]"
        assert type(kwargs["center_frequency"]) is int, "center_frequency must be a int"
        assert type(kwargs["sampling_frequency"]) is int, "sampling_frequency must be a int"
        assert type(kwargs["ax_win_size"]) is float, "ax_win_size must be a float"
        assert type(kwargs["lat_win_size"]) is float, "lat_win_size must be a float"
        assert type(kwargs["window_thresh"]) is float, "window_thresh must be a float"
        assert type(kwargs["axial_overlap"]) is float, "axial_overlap must be a float"
        assert type(kwargs["lateral_overlap"]) is float, "lateral_overlap must be a float"

        out.transducer_freq_band = kwargs["transducer_freq_band"]
        out.analysis_freq_band = kwargs["analysis_freq_band"]
        out.center_frequency = kwargs["center_frequency"]
        out.sampling_frequency = kwargs["sampling_frequency"]
        out.ax_win_size = kwargs["ax_win_size"]
        out.lat_win_size = kwargs["lat_win_size"]
        out.window_thresh = kwargs["window_thresh"]
        out.axial_overlap = kwargs["axial_overlap"]
        out.lateral_overlap = kwargs["lateral_overlap"]
        out.cor_win_size = kwargs.get("cor_win_size", None)
        out.coronal_overlap = kwargs.get("coronal_overlap", None)

        if out.cor_win_size is not None:
            assert type(out.cor_win_size) is float, "cor_win_size must be a float"
            assert type(out.coronal_overlap) is float, "coronal_overlap must be a float"
            
    except KeyError as e:
        raise KeyError(f"Missing required key: {e}. Please provide all necessary parameters for the custom configuration.")

    return out