# transformation_mirror.py

import cv2
import numpy as np

def mirror_image(image: np.ndarray, horizontal: bool = True, vertical: bool = True) -> np.ndarray:
    """
    Mirrors the image along the specified axes.
    
    Parameters:
        image (np.ndarray): Input image to be mirrored.
        horizontal (bool): If True, mirror top to bottom (flip vertically).
        vertical (bool): If True, mirror left to right (flip horizontally).
        
    Returns:
        np.ndarray: Transformed image.
    """
    if horizontal and vertical:
        return cv2.flip(image, -1)  # Flip both axes
    elif horizontal:
        return cv2.flip(image, 0)   # Flip vertically
    elif vertical:
        return cv2.flip(image, 1)   # Flip horizontally
    else:
        return image.copy()         # No flip

