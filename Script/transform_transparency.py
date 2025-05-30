import tkinter as tk
from tkinter import ttk
from overlay_display import apply_transparency
from buttons import create_button_row

class TransparencyControlFrame(ttk.Frame):
    def __init__(self, parent, on_change_callback):
        super().__init__(parent, padding="5")
        self.on_change_callback = on_change_callback

        self.transparency_var = tk.IntVar(value=70)

        self.build_ui()

    def build_ui(self):
        ttk.Label(self, text="Transparency (0% to 100%)").pack(anchor=tk.W)

        control_frame = ttk.Frame(self)
        control_frame.pack(pady=(5, 0), fill=tk.X)

        from buttons import create_button_row

        create_button_row(self, self.transparency_var, [-10, -1, 1, 10], self.on_change_callback)

    def get_transparency(self):
        return self.transparency_var.get()

    def reset(self):
        self.transparency_var.set(70)
        self.on_change_callback()
