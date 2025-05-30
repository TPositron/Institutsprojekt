import tkinter as tk

class CanvasMoveControl:
    def __init__(self, canvas):
        self.canvas = canvas
        self.start_x = 0
        self.start_y = 0
        self.dragging = False
        
        # Bind mouse events for panning
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan_image)
        self.canvas.bind("<ButtonRelease-1>", self.stop_pan)
        
        # Bind mouse wheel for scrolling
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.canvas.bind("<Button-4>", self.mouse_wheel)  # Linux
        self.canvas.bind("<Button-5>", self.mouse_wheel)  # Linux
        
        # Bind arrow keys for keyboard navigation
        self.canvas.bind("<Key>", self.key_pan)
        self.canvas.focus_set()  # Make canvas focusable
        
    def start_pan(self, event):
        """Start panning operation"""
        self.canvas.scan_mark(event.x, event.y)
        self.start_x = event.x
        self.start_y = event.y
        self.dragging = True
        self.canvas.config(cursor="fleur")  # Change cursor to indicate dragging
        
    def pan_image(self, event):
        """Pan the image"""
        if self.dragging:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            
    def stop_pan(self, event):
        """Stop panning operation"""
        self.dragging = False
        self.canvas.config(cursor="")  # Reset cursor
        
    def mouse_wheel(self, event):
        """Handle mouse wheel scrolling"""
        # Vertical scrolling
        if event.delta:
            delta = -1 * (event.delta / 120)
        elif event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            delta = 0
            
        # Check if Shift is held for horizontal scrolling
        if event.state & 0x1:  # Shift key
            self.canvas.xview_scroll(int(delta), "units")
        else:
            self.canvas.yview_scroll(int(delta), "units")
            
    def key_pan(self, event):
        """Handle keyboard panning"""
        pan_distance = 50
        
        if event.keysym == "Left":
            self.canvas.xview_scroll(-1, "units")
        elif event.keysym == "Right":
            self.canvas.xview_scroll(1, "units")
        elif event.keysym == "Up":
            self.canvas.yview_scroll(-1, "units")
        elif event.keysym == "Down":
            self.canvas.yview_scroll(1, "units")
            
    def center_image(self):
        """Center the image in the canvas"""
        self.canvas.xview_moveto(0.5)
        self.canvas.yview_moveto(0.5)
        
    def reset_view(self):
        """Reset view to top-left corner"""
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)