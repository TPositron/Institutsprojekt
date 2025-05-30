import tkinter as tk
from tkinter import ttk

def create_button_row(parent, variable, steps, callback):
    """
    Create a row of --, -, Entry, +, ++ buttons.

    Args:
        parent: parent frame
        variable: associated tk.IntVar or tk.DoubleVar
        steps: list like [-10, -1, 1, 10]
        callback: function to call when changed
    """
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=(0, 10))

    for step in steps[:2]:  # negative buttons
        ttk.Button(frame, text="--" if step < -1 else "-", width=3,
                   command=lambda s=step: adjust_var(variable, s, callback)).pack(side=tk.LEFT)

    entry = ttk.Entry(frame, textvariable=variable, width=6)
    entry.pack(side=tk.LEFT, padx=5)
    entry.bind('<Return>', lambda e: callback())

    for step in steps[2:]:  # positive buttons
        ttk.Button(frame, text="++" if step > 1 else "+", width=3,
                   command=lambda s=step: adjust_var(variable, s, callback)).pack(side=tk.LEFT)

    return frame

def adjust_var(var, delta, callback):
    try:
        new_val = var.get() + delta
        var.set(new_val)
    except tk.TclError:
        pass
    callback()
