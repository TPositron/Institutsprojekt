import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from buttons import create_button_row

def rotate_image(image, angle_degrees):
    """
    Rotate the image around its center with edge replication.

    Args:
        image: Input grayscale image (numpy array)
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)

    Returns:
        Rotated image
    """
    if not (-90 <= angle_degrees <= 90):
        raise ValueError("Rotation angle must be between -90 and +90 degrees")

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

class RotateTransformFrame(ttk.Frame):
    def __init__(self, parent, on_change_callback):
        super().__init__(parent, padding="5")
        self.on_change_callback = on_change_callback

        self.rotation_var = tk.DoubleVar(value=0.0)

        self.create_controls()
        
    def create_controls(self):
        ttk.Label(self, text="Rotation (-90° to +90°):").pack(anchor=tk.W)
        create_button_row(self, self.rotation_var, [-10.0, -1.0, 1.0, 10.0], self.on_change_callback) 

    def get_angle(self):
        return self.rotation_var.get()

    def reset(self):
        self.rotation_var.set(0.0)
        self.on_change_callback()
