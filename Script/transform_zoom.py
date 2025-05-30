import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from buttons import create_button_row

def zoom_image(image, zoom_percent):
    """
    Zoom the image around its center.
    Args:
        image: Grayscale image (numpy array).
        zoom_percent: Zoom level (e.g., 100 = normal, >100 = zoom in, <100 = zoom out)
    Returns:
        Transformed image with white padding.
    """
    if zoom_percent <= 0:
        raise ValueError("Zoom percentage must be > 0")

    h, w = image.shape[:2]
    scale = zoom_percent / 100.0
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


class ZoomTransformFrame(ttk.Frame):
    def __init__(self, parent, on_change_callback):
        super().__init__(parent, padding="5")
        self.on_change_callback = on_change_callback

        # Variable
        self.zoom_var = tk.IntVar(value=100)

        # UI
        self.create_controls()

    def create_controls(self):
        ttk.Label(self, text="Zoom (10% to 300%):").pack(anchor=tk.W)
        create_button_row(self, self.zoom_var, [-10, -1, 1, 10], self.on_change_callback)

    def get_zoom(self):
        return self.zoom_var.get()

    def reset(self):
        self.zoom_var.set(100)
        self.on_change_callback()
