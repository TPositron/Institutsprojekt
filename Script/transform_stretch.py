import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from buttons import create_button_row

def stretch_image(image, x_scale, y_scale):
    """
    Stretch the image in x and y directions around the center with white padding.

    Args:
        image: Grayscale image (numpy array)
        x_scale: Horizontal scale factor (1.0 = no stretch)
        y_scale: Vertical scale factor (1.0 = no stretch)

    Returns:
        Stretched image (same dimensions as input)
    """
    h, w = image.shape[:2]
    M = np.float32([
        [x_scale, 0, w / 2 * (1 - x_scale)],
        [0, y_scale, h / 2 * (1 - y_scale)]
    ])
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)

class StretchTransformFrame(ttk.Frame):
    def __init__(self, parent, on_change_callback):
        super().__init__(parent, padding="5")
        self.on_change_callback = on_change_callback

        self.stretch_x_var = tk.DoubleVar(value=1.0)
        self.stretch_y_var = tk.DoubleVar(value=1.0)

        self.create_controls()
           
    def create_controls(self):
        ttk.Label(self, text="Stretch X (0.1 to 3.0):").pack(anchor=tk.W)
        create_button_row(self, self.stretch_x_var, [-0.1, -0.01, 0.01, 0.1], self.on_change_callback)
    
        ttk.Label(self, text="Stretch Y (0.1 to 3.0):").pack(anchor=tk.W)
        create_button_row(self, self.stretch_y_var, [-0.1, -0.01, 0.01, 0.1], self.on_change_callback)

    def get_scales(self):
        return self.stretch_x_var.get(), self.stretch_y_var.get()

    def reset(self):
        self.stretch_x_var.set(1.0)
        self.stretch_y_var.set(1.0)
        self.on_change_callback()
