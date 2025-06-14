"""
Utility functions for the MBES pipe detection module.
"""

import cv2
import numpy as np
from scipy.interpolate import griddata

def pcl_buffer_to_intensity(pcl_buffer, resolution):
    """
    Convert pcl buffer of shape (num_pings, num_points, 4) to an intensity image.
    The intensity image will have shape (num_y_pixels, num_x_pixels) where
    num_x_pixels and num_y_pixels are determined by the resolution and the range of x and y coordinates.

    Returns:
        dict: A dictionary containing:
            - 'intensity_image': The interpolated intensity image.
            - 'mask': A boolean mask indicating valid pixels in the intensity image.
        if pcl_buffer is None or has insufficient data, returns None.
    """
    if pcl_buffer is None:
        return None
    x = pcl_buffer[:, :, 0]
    y = pcl_buffer[:, :, 1]
    z = pcl_buffer[:, :, 2]
    intensities = pcl_buffer[:, :, 3]

    x_min, x_max = (np.min(x), np.max(x))
    y_min, y_max = (np.min(y), np.max(y))
    num_x_pixels = int((x_max - x_min) / resolution)
    num_y_pixels = int((y_max - y_min) / resolution)

    if num_x_pixels <= 0 or num_y_pixels <= 0:
        return None

    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, num_x_pixels),
        np.linspace(y_min, y_max, num_y_pixels)
    )
    intensity_image = griddata(
        (x.flatten(), y.flatten()),
        intensities.flatten(),
        (X, Y),
        method='linear',
    )
    mask = ~np.isnan(intensity_image)
    return {
        'intensity_image': intensity_image,
        'mask': mask,
    }


def normalize_intensity_image(intensity_image):
    """
    Normalize the intensity image to the range [0, 255].
    """
    normalized_image = cv2.normalize(
        intensity_image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    ).astype(np.uint8)
    return normalized_image