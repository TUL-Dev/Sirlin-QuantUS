from typing import List

class RfAnalysisConfig:
    """
    Class to store configuration data for RF analysis.
    """

    def __init__(self):
        self.transducer_freq_band: List[int]  # [min, max] (Hz)
        self.analysis_freq_band: List[int]  # [lower, upper] (Hz)
        self.sampling_frequency: int  # Hz
        self.ax_win_size: float  # axial length per window (mm)
        self.lat_win_size: float  # lateral width per window (mm)
        self.window_thresh: float  # % of window area required to be in ROI
        self.axial_overlap: float  # % of ax window length to move before next window
        self.lateral_overlap: float  # % of lat window length to move before next window
        self.center_frequency: float  # Hz
        
        # 3D scan parameters
        self.cor_win_size: float  # coronal length per window (mm)
        self.coronal_overlap: float # % of cor window length to move before next window
