from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ...image_loading.utc_loaders.transforms import scanConvert
import cv2

from ...data_objs.analysis import ParamapAnalysisBase

def plot_ps_window_data(analysis_obj: ParamapAnalysisBase, dest_folder: str) -> None:
    """Plots the power spectrum data for each window in the ROI.

    The power spectrum data is plotted along with the average power spectrum and a line of best fit
    used for the midband fit, spectral slope, and spectral intercept calculations. Also plots the 
    frequency band used for analysis.
    """
    assert Path(dest_folder).is_dir(), "plot_ps_data visualization: Power spectrum plot output folder doesn't exist"
    assert hasattr(analysis_obj.windows[0].results, 'ss'), "Spectral slope not found in results"
    assert hasattr(analysis_obj.windows[0].results, 'si'), "Spectral intercept not found in results"
    assert hasattr(analysis_obj.windows[0].results, 'nps'), "Normalized power spectrum not found in results"
    assert hasattr(analysis_obj.windows[0].results, 'f'), "Frequency not found in results"
    
    ss_arr = [window.results.ss for window in analysis_obj.windows]
    si_arr = [window.results.si for window in analysis_obj.windows]
    nps_arr = [window.results.nps for window in analysis_obj.windows]

    fig, ax = plt.subplots()

    ss_mean = np.mean(np.array(ss_arr)/1e6)
    si_mean = np.mean(si_arr)
    nps_arr = [window.results.nps for window in analysis_obj.windows]
    av_nps = np.mean(nps_arr, axis=0)
    f = analysis_obj.windows[0].results.f
    x = np.linspace(min(f), max(f), 100)
    y = ss_mean*x + si_mean

    for nps in nps_arr[:-1]:
        ax.plot(f/1e6, nps, c="b", alpha=0.2)
    ax.plot(f/1e6, nps_arr[-1], c="b", alpha=0.2, label="Window NPS")
    ax.plot(f/1e6, av_nps, color="r", label="Av NPS")
    ax.plot(x/1e6, y, c="orange", label="Av LOBF")
    ax.plot(2*[analysis_obj.config.analysis_freq_band[0]/1e6], [np.amin(nps_arr), np.amax(nps_arr)], c="purple")
    ax.plot(2*[analysis_obj.config.analysis_freq_band[1]/1e6], [np.amin(nps_arr), np.amax(nps_arr)], c="purple", label="Analysis Band")
    ax.set_title("Normalized Power Spectra")
    ax.legend()
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power (dB)")
    ax.set_ylim([np.amin(nps_arr), np.amax(nps_arr)])
    ax.set_xlim([min(f)/1e6, max(f)/1e6])
    
    fig.savefig(Path(dest_folder) / 'nps_plot.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_hscan_data(analysis_obj: ParamapAnalysisBase, dest_folder: str) -> None:
    """Plots the H-scan results overlaid on the B-mode image.

    The H-scan results consist of two channels (red and blue) that represent different
    frequency components and scatterer properties in the ultrasound signal. These are overlaid
    on the B-mode image to provide anatomical context. Red channel represents low frequency
    components while blue channel represents high frequency components.
    """
    assert Path(dest_folder).is_dir(), "plot_hscan_data visualization: H-scan plot output folder doesn't exist"
    assert hasattr(analysis_obj.single_window.results, 'hscan_blue_channel'), "H-scan blue channel not found in results"
    assert hasattr(analysis_obj.single_window.results, 'hscan_red_channel'), "H-scan red channel not found in results"
    
    # Get the B-mode image and convert to RGB
    if analysis_obj.image_data.sc_bmode is not None:
        bmode = cv2.cvtColor(np.array(analysis_obj.image_data.sc_bmode, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        bmode = cv2.cvtColor(np.array(analysis_obj.image_data.bmode, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
    
    # Get H-scan results
    blue_channel = analysis_obj.single_window.results.hscan_blue_channel
    red_channel = analysis_obj.single_window.results.hscan_red_channel
    
    # Create window index map
    window_idx_map = np.zeros_like(analysis_obj.image_data.bmode, dtype=int)
    
    # Fill window index map using all windows
    for i, window in enumerate(analysis_obj.windows):
        if window.cor_min == -1:
            window_idx_map[window.ax_min:window.ax_max+1, window.lat_min:window.lat_max+1] = i+1
        else:
            window_idx_map[window.cor_min:window.cor_max+1, window.lat_min:window.lat_max+1, window.ax_min:window.ax_max+1] = i+1
    
    # Create full-size channel images with ROI data
    blue_full = np.zeros_like(analysis_obj.image_data.bmode, dtype=np.float32)
    red_full = np.zeros_like(analysis_obj.image_data.bmode, dtype=np.float32)
    
    # Get single window boundaries
    single_window = analysis_obj.single_window
    
    # Apply the window mask to the channel data using window_idx_map
    window_points = np.transpose(np.where(window_idx_map > 0))
    for point in window_points:
        # Map from global coordinates to single window coordinates
        rel_ax = point[0] - single_window.ax_min
        rel_lat = point[1] - single_window.lat_min
        if 0 <= rel_ax < blue_channel.shape[0] and 0 <= rel_lat < blue_channel.shape[1]:
            blue_full[tuple(point)] = blue_channel[rel_ax, rel_lat]
            red_full[tuple(point)] = red_channel[rel_ax, rel_lat]
    
    # Calculate normalization ranges using percentiles to handle outliers
    # Only consider values within the ROI for normalization
    valid_blue = blue_full[window_idx_map > 0]
    valid_red = red_full[window_idx_map > 0]
    blue_min, blue_max = np.percentile(valid_blue[valid_blue != 0], [5, 95])
    red_min, red_max = np.percentile(valid_red[valid_red != 0], [5, 95])
    
    # Scan convert all the data
    if analysis_obj.image_data.sc_bmode is not None:
        image_data = analysis_obj.image_data
        if image_data.sc_bmode.ndim == 2:
            # First scan convert the window index map
            sc_window_idx_map = np.array(scanConvert(window_idx_map, image_data.width,
                                                   image_data.tilt, image_data.start_depth,
                                                   image_data.end_depth, desiredHeight=image_data.sc_bmode.shape[0])[0].scArr,
                                       dtype=int)
            
            # Then scan convert the channel data
            sc_blue = scanConvert(blue_full, image_data.width, image_data.tilt,
                                image_data.start_depth, image_data.end_depth,
                                desiredHeight=image_data.sc_bmode.shape[0])[0].scArr
            
            sc_red = scanConvert(red_full, image_data.width, image_data.tilt,
                               image_data.start_depth, image_data.end_depth,
                               desiredHeight=image_data.sc_bmode.shape[0])[0].scArr
            
            # Use the scan-converted window map as the mask
            sc_mask = (sc_window_idx_map > 0).astype(np.float32)
        else:
            raise NotImplementedError("3D scan conversion not yet implemented for H-scan visualization")
    else:
        sc_blue = blue_full
        sc_red = red_full
        sc_mask = (window_idx_map > 0).astype(np.float32)
    
    # Normalize each channel
    sc_blue_norm = np.clip((sc_blue - blue_min) / (blue_max - blue_min), 0, 1)
    sc_red_norm = np.clip((sc_red - red_min) / (red_max - red_min), 0, 1)
    
    # Create RGBA overlay
    overlay = np.zeros((*bmode.shape[:2], 4), dtype=np.float32)
    
    # Set RGB values where mask is active
    overlay[..., 0] = sc_red_norm * sc_mask  # Red channel
    overlay[..., 2] = sc_blue_norm * sc_mask  # Blue channel
    overlay[..., 3] = sc_mask * 0.5  # Alpha channel (50% transparency where mask is active)
    
    # Calculate proper aspect ratio based on resolution
    if analysis_obj.image_data.sc_bmode is not None:
        width = bmode.shape[1] * analysis_obj.image_data.sc_lateral_res
        height = bmode.shape[0] * analysis_obj.image_data.sc_axial_res
    else:
        width = bmode.shape[1] * analysis_obj.image_data.lateral_res
        height = bmode.shape[0] * analysis_obj.image_data.axial_res
    aspect = width/height
    
    # Create figure for main overlay
    fig, ax = plt.subplots()
    
    # Plot B-mode image and overlay
    im = ax.imshow(bmode)
    im = ax.imshow(overlay)
    extent = im.get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
    ax.axis('off')
    
    # Save the main overlay figure without any padding
    fig.savefig(Path(dest_folder) / 'hscan_overlay.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # Create and save red channel legend
    fig_red, ax_red = plt.subplots(figsize=(2, 10))
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax_red.imshow(gradient, aspect='auto', cmap=plt.cm.Reds)
    ax_red.set_yticks(np.linspace(0, 255, 6))
    ax_red.set_yticklabels([f"{np.round(red_min + i*((red_max-red_min)/5), 2)}" for i in np.linspace(0, 5, 6)])
    ax_red.spines['top'].set_visible(False)
    ax_red.spines['right'].set_visible(False)
    ax_red.spines['left'].set_visible(False)
    ax_red.spines['bottom'].set_visible(False)
    ax_red.set_xticks([])
    ax_red.set_title('Low Frequency')
    ax_red.invert_yaxis()
    fig_red.savefig(Path(dest_folder) / 'hscan_red_legend.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig_red)
    
    # Create and save blue channel legend
    fig_blue, ax_blue = plt.subplots(figsize=(2, 10))
    ax_blue.imshow(gradient, aspect='auto', cmap=plt.cm.Blues)
    ax_blue.set_yticks(np.linspace(0, 255, 6))
    ax_blue.set_yticklabels([f"{np.round(blue_min + i*((blue_max-blue_min)/5), 2)}" for i in np.linspace(0, 5, 6)])
    ax_blue.spines['top'].set_visible(False)
    ax_blue.spines['right'].set_visible(False)
    ax_blue.spines['left'].set_visible(False)
    ax_blue.spines['bottom'].set_visible(False)
    ax_blue.set_xticks([])
    ax_blue.set_title('High Frequency')
    ax_blue.invert_yaxis()
    fig_blue.savefig(Path(dest_folder) / 'hscan_blue_legend.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig_blue)



