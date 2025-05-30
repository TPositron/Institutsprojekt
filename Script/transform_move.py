import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from buttons import create_button_row

def move_image(image, dx, dy):
    """Move image in x and y directions with white padding."""
    h, w = image.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return moved

class MoveTransformFrame(ttk.Frame):
    def __init__(self, parent, on_change_callback):
        super().__init__(parent, padding="5")
        self.on_change_callback = on_change_callback

        # Variables
        self.move_x_var = tk.IntVar(value=0)
        self.move_y_var = tk.IntVar(value=0)
        
        self.create_controls()

    def create_controls(self):
        ttk.Label(self, text="Move X:").pack(anchor=tk.W)
        create_button_row(self, self.move_x_var, [-10, -1, 1, 10], self.on_change_callback)
        
        ttk.Label(self, text="Move Y:").pack(anchor=tk.W)
        create_button_row(self, self.move_y_var, [-10, -1, 1, 10], self.on_change_callback)

    def get_params(self):
        return self.move_x_var.get(), self.move_y_var.get()

    def reset(self):
        self.move_x_var.set(0)
        self.move_y_var.set(0)
        self.on_change_callback()
