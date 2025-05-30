import tkinter as tk
from tkinter import ttk

class CanvasZoomControl(ttk.Frame):
    def __init__(self, parent, on_zoom_change):
        super().__init__(parent, padding="5")
        self.on_zoom_change = on_zoom_change

        # Zoom level (1.0 = 100%)
        self.zoom_var = tk.DoubleVar(value=1.0)

        # Preset buttons
        ttk.Label(self, text="Canvas Zoom:").grid(row=0, column=0, sticky=tk.W, columnspan=6)

        presets = [("50%", 0.5), ("75%", 0.75), ("100%", 1.0), ("125%", 1.25), ("150%", 1.5)]
        for i, (label, value) in enumerate(presets):
            ttk.Button(self, text=label, command=lambda v=value: self.set_zoom(v)).grid(row=1, column=i)

        # Slider (0.1 to 3.0 scale)
        ttk.Scale(self, from_=0.1, to=3.0, orient=tk.HORIZONTAL, variable=self.zoom_var,
                  command=lambda e: self.on_zoom_change()).grid(row=2, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=5)

    def set_zoom(self, value):
        self.zoom_var.set(value)
        self.on_zoom_change()

    def get_zoom(self):
        return self.zoom_var.get()

    def bind_mouse_wheel(self, widget):
        widget.bind("<MouseWheel>", self._on_mouse_wheel)
        widget.bind("<Button-4>", self._on_mouse_wheel)  # Linux
        widget.bind("<Button-5>", self._on_mouse_wheel)

    def _on_mouse_wheel(self, event):
        delta = 0.1 if (event.delta > 0 or event.num == 4) else -0.1
        self.adjust_zoom(delta)

    def bind_keyboard_shortcuts(self, widget):
        widget.bind("<Control-plus>", lambda e: self.adjust_zoom(0.1))
        widget.bind("<Control-minus>", lambda e: self.adjust_zoom(-0.1))

    def adjust_zoom(self, delta):
        new_zoom = self.zoom_var.get() + delta
        self.zoom_var.set(max(0.1, min(3.0, new_zoom)))
        self.on_zoom_change()
