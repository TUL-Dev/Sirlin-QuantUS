# Standard Library Imports
from __future__ import annotations

# Third-Party Library Imports
import os
import numpy as np
from pathlib import Path

# Local Module Imports
from ....data_objs.image import UltrasoundRfImage
from .parser import ClariusTarUnpacker, clariusRfParserWrapper

class EntryClass(UltrasoundRfImage):
    """
    Class for Clarius RF image data.
    This class is used to parse RF data from Clarius ultrasound machines.
    Supports both raw RF files and compressed .tar files.
    """
    extensions = [".raw", ".tar"]  # Clarius RF files can be raw or tar compressed
    spatial_dims = 2  # Clarius data can be 3D but we extract 2D frames
    
    def __init__(self, scan_path: str, phantom_path: str, **kwargs):
        """Initialize the Clarius RF loader.
        
        Args:
            scan_path (str): Path to the scan RF file (.raw or .tar)
            phantom_path (str): Path to the phantom RF file (.raw or .tar)
            **kwargs:
                visualize (bool): Whether to visualize the data in the terminal after extraction.
                use_tgc (bool): Whether to use TGC data in processing.
        """
        super().__init__(scan_path, phantom_path)

        visualize = kwargs.get('visualize', False)
        use_tgc = kwargs.get('use_tgc', True)

        # Validate file extensions
        scan_path = Path(scan_path)
        phantom_path = Path(phantom_path)
        assert scan_path.suffix in self.extensions, f"Scan file must end with {self.extensions}"
        assert phantom_path.suffix in self.extensions, f"Phantom file must end with {self.extensions}"
        
        # Handle tar files if provided and get extracted folder paths
        scan_raw_dir = self._handle_tar_file(str(scan_path))
        phantom_raw_dir = self._handle_tar_file(str(phantom_path))
                
        # Parse the data using the Clarius parser
        imgData, imgInfo, refData, refInfo, scanConverted = clariusRfParserWrapper(
            os.path.dirname(scan_raw_dir),
            os.path.dirname(phantom_raw_dir),
            visualize=visualize,
            use_tgc=use_tgc,
        )
        
        # Set the required attributes
        frame_idx = kwargs.get('ref_frame_idx', 0)
        
        # Extract the frame and ensure correct orientation
        self.rf_data = imgData.rf  # Shape will be (frames, axial, lateral)
        self.phantom_rf_data = refData.rf[frame_idx]
        self.bmode = imgData.bMode
        
        # Calculate resolutions
        self.axial_res = imgInfo.yResRF  # Use the RF resolution directly
        self.lateral_res = imgInfo.xResRF
        
        # Scan conversion data if necessary
        if scanConverted:
            self.sc_bmode = imgData.scBmode if imgData.scBmode is not None else None
            self.sc_axial_res = imgInfo.axialRes
            self.sc_lateral_res = imgInfo.lateralRes
            self.xmap = imgData.scBmodeStruct.xmap
            self.ymap = imgData.scBmodeStruct.ymap
            self.tilt = imgInfo.tilt1
            self.width = imgInfo.width1
            self.start_depth = imgInfo.startDepth1
            self.end_depth = imgInfo.endDepth1
            
    def _handle_tar_file(self, file_path: str) -> str:
        """Handle tar file extraction if needed.
        
        Args:
            file_path (str): Path to the file to check
            
        Returns:
            str: Path to the raw file (either original path if raw or extracted path if tar)
        """
        if Path(file_path).suffix == '.tar':
            # Extract the tar file
            unpacker = ClariusTarUnpacker(file_path, extraction_mode='single_tar')
            
            # Get the extracted folder path
            extracted_folder = os.path.join(
                os.path.dirname(file_path),
                f"{Path(file_path).stem}_extracted"
            )
            
            # Find the rf.raw file in the extracted folder
            for root, _, files in os.walk(extracted_folder):
                for file in files:
                    if file.endswith('rf.raw'):
                        return os.path.join(root, file)
            
            raise FileNotFoundError(f"No rf.raw file found in extracted folder: {extracted_folder}")
        
        return file_path
