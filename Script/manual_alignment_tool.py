import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image, ImageTk

# Import your modules
from buttons import create_button_row
from canvas_zoom import CanvasZoomControl
from overlay_display import OverlayComposer, create_tkinter_photo
from transform_move import move_image, MoveTransformFrame
from transform_rotate import rotate_image, RotateTransformFrame
from transform_stretch import stretch_image, StretchTransformFrame
from transform_zoom import zoom_image, ZoomTransformFrame
from transform_transparency import TransparencyControlFrame
from transformation_mirror import mirror_image


class ManualAlignmentUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Manual Alignment Tool")
        self.root.geometry("1400x1400")
        
        # Data paths
        self.gds_object_dir = "C:\\Users\\tarik\Desktop\\Bildanalyse\\Results\\exact_GDS\\Optimized"
        self.sem_cropped_dir = "C:\\Users\\tarik\Desktop\\Bildanalyse\\Results\\Centers\\SEM\\cropped"
        self.output_dir = "C:\\Users\\tarik\\Desktop\\Bildanalyse\\Results\\Aligned"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Image data
        self.original_sem = None
        self.original_gds = None
        self.current_gds = None
        # Initialize current_file_pair
        self.current_file_pair = None
        
        # Canvas and display
        self.canvas_size = (1024, 1024)
        self.overlay_composer = OverlayComposer()
        
        # Transformation parameters
        self.transform_params = {
            'move_x': 0, 'move_y': 0,
            'rotation': 0.0,
            'stretch_x': 1.0, 'stretch_y': 1.0,
            'zoom': 100,
            'transparency': 40,
            'mirror_horizontal': False,
            'mirror_vertical': False
        }
        self.setup_ui()
           
    def setup_ui(self):
        """Setup the main UI layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel - Canvas
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_control_panel(left_panel)
        self.setup_canvas_panel(right_panel)
        
    def setup_control_panel(self, parent):
        
        file_frame = ttk.LabelFrame(parent, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        # Manual file selection buttons
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Select SEM", 
                  command=self.select_sem_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Select GDS", 
                  command=self.select_gds_file).pack(side=tk.LEFT)
        
        # Transformation controls
        transform_frame = ttk.LabelFrame(parent, text="Transformations", padding="10")
        transform_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Move controls
        move_frame = ttk.LabelFrame(transform_frame, text="Move", padding="5")
        move_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(move_frame, text="Move X:").pack(anchor=tk.W)
        self.move_x_var = tk.IntVar(value=0)
        create_button_row(move_frame, self.move_x_var, [-10, -1, 1, 10], self.update_transform)
        
        ttk.Label(move_frame, text="Move Y:").pack(anchor=tk.W)
        self.move_y_var = tk.IntVar(value=0)
        create_button_row(move_frame, self.move_y_var, [-10, -1, 1, 10], self.update_transform)
        
        # Rotation controls
        rotate_frame = ttk.LabelFrame(transform_frame, text="Rotation", padding="5")
        rotate_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(rotate_frame, text="Rotation (-90° to +90°):").pack(anchor=tk.W)
        self.rotation_var = tk.DoubleVar(value=0.0)
        create_button_row(rotate_frame, self.rotation_var, [-10.0, -1.0, 1.0, 10.0], self.update_transform)
        
        # Stretch controls
        stretch_frame = ttk.LabelFrame(transform_frame, text="Stretch", padding="5")
        stretch_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(stretch_frame, text="Stretch X (0.1 to 3.0):").pack(anchor=tk.W)
        self.stretch_x_var = tk.DoubleVar(value=1.0)
        create_button_row(stretch_frame, self.stretch_x_var, [-0.1, -0.01, 0.01, 0.1], self.update_transform)
        
        ttk.Label(stretch_frame, text="Stretch Y (0.1 to 3.0):").pack(anchor=tk.W)
        self.stretch_y_var = tk.DoubleVar(value=1.0)
        create_button_row(stretch_frame, self.stretch_y_var, [-0.1, -0.01, 0.01, 0.1], self.update_transform)
        
        # Zoom controls
        zoom_frame = ttk.LabelFrame(transform_frame, text="Zoom", padding="5")
        zoom_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(zoom_frame, text="Zoom (10% to 300%):").pack(anchor=tk.W)
        self.zoom_var = tk.IntVar(value=100)
        create_button_row(zoom_frame, self.zoom_var, [-10, -1, 1, 10], self.update_transform)
        
        # Transparency controls
        trans_frame = ttk.LabelFrame(transform_frame, text="Transparency", padding="5")
        trans_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(trans_frame, text="Transparency (0% to 100%):").pack(anchor=tk.W)
        self.transparency_var = tk.IntVar(value=70)
        create_button_row(trans_frame, self.transparency_var, [-10, -1, 1, 10], self.update_transform)
        
         # Mirror controls
        mirror_frame = ttk.LabelFrame(transform_frame, text="Mirror", padding="5")
        mirror_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(mirror_frame, text="Horizontal", command=self.toggle_mirror_horizontal).pack(side=tk.LEFT, padx=5)
        ttk.Button(mirror_frame, text="Vertical", command=self.toggle_mirror_vertical).pack(side=tk.LEFT, padx=5)

        
        # Action buttons
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(action_frame, text="Reset All", 
                  command=self.reset_transforms).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(action_frame, text="Save Transformation", 
                  command=self.save_transformation).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(action_frame, text="Load Transformation", 
                  command=self.load_transformation).pack(fill=tk.X)
        
    def setup_canvas_panel(self, parent):
        """Setup the canvas panel with zoom controls"""
        # Canvas zoom controls
        self.canvas_zoom = CanvasZoomControl(parent, self.on_canvas_zoom_change)
        self.canvas_zoom.pack(fill=tk.X, pady=(0, 10))
        
        # Canvas with scrollbars
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg='white', 
                               width=self.canvas_size[0], height=self.canvas_size[1])
        
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        # Pack scrollbars and canvas
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mouse events for zoom - fix the binding
        self.canvas_zoom.bind_mouse_wheel(self.canvas)
        self.canvas_zoom.bind_keyboard_shortcuts(self.root)  # Bind to root instead
        
        # Make canvas focusable for keyboard shortcuts
        self.canvas.focus_set()
        
    def on_canvas_zoom_change(self):
        """Handle canvas zoom changes"""
        self.update_display()
        
    def select_sem_file(self):
        """Load SEM image"""
        filename = filedialog.askopenfilename(
            title="Select SEM Image",
            initialdir=self.sem_cropped_dir, #Anfang in sem Directory
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tiff *.bmp")]
        )
        if filename:
            self.original_sem = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if self.original_sem is None:
                messagebox.showerror("Error", "Failed to load SEM image")
            else:
                base_name = os.path.splitext(os.path.basename(filename))[0]
                self.current_file_pair = base_name
                messagebox.showinfo("Success", f"Loaded SEM: {base_name}")
                self.update_display()

    def select_gds_file(self):
        """Load GDS image"""
        filename = filedialog.askopenfilename(
            title="Select GDS Image",
            initialdir=self.gds_object_dir, #start in gds directory
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tiff *.bmp")]
        )
        if filename:
            self.original_gds = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if self.original_gds is None:
                messagebox.showerror("Error", "Failed to load GDS image")
            else:
                self.current_gds = self.original_gds.copy()
                if not self.current_file_pair:
                    base_name = os.path.splitext(os.path.basename(filename))[0]
                    self.current_file_pair = base_name
                messagebox.showinfo("Success", "Loaded GDS image")
                self.reset_transforms()

    def update_transform(self):
        """Update transformation parameters and apply to GDS image"""
        if self.original_gds is None:
            return
        
        try:
            # Get current parameter values
            self.transform_params = {
                'move_x': self.move_x_var.get(),
                'move_y': self.move_y_var.get(),
                'rotation': max(-90, min(90, self.rotation_var.get())),
                'stretch_x': max(0.1, min(3.0, self.stretch_x_var.get())),
                'stretch_y': max(0.1, min(3.0, self.stretch_y_var.get())),
                'zoom': max(10, min(300, self.zoom_var.get())),
                'transparency': max(0, min(100, self.transparency_var.get())),
                'mirror_horizontal': self.transform_params.get('mirror_horizontal', False),
                'mirror_vertical': self.transform_params.get('mirror_vertical', False)
            }
            
            # Apply transformations sequentially
            transformed = self.original_gds.copy()
            
            # 1. Zoom
            if self.transform_params['zoom'] != 100:
                transformed = zoom_image(transformed, self.transform_params['zoom'])
            
            # 2. Stretch
            if self.transform_params['stretch_x'] != 1.0 or self.transform_params['stretch_y'] != 1.0:
                transformed = stretch_image(transformed, 
                                          self.transform_params['stretch_x'],
                                          self.transform_params['stretch_y'])
            
            # 3. Rotate
            if self.transform_params['rotation'] != 0.0:
                transformed = rotate_image(transformed, self.transform_params['rotation'])
            
            # 4. Move
            if self.transform_params['move_x'] != 0 or self.transform_params['move_y'] != 0:
                transformed = move_image(transformed, 
                                       self.transform_params['move_x'],
                                       self.transform_params['move_y'])
            # 5. Mirror
            if self.transform_params.get('mirror_horizontal') or self.transform_params.get('mirror_vertical'):
                transformed = mirror_image(transformed,
                                           horizontal=self.transform_params['mirror_horizontal'],
                                           vertical=self.transform_params['mirror_vertical'])

            self.current_gds = transformed
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Transformation failed: {str(e)}")
    
    def toggle_mirror_horizontal(self):
        """Toggle horizontal mirror and update image."""
        self.transform_params['mirror_horizontal'] = not self.transform_params['mirror_horizontal']
        self.update_transform()

    def toggle_mirror_vertical(self):
        """Toggle vertical mirror and update image."""
        self.transform_params['mirror_vertical'] = not self.transform_params['mirror_vertical']
        self.update_transform()

    def update_display(self):
        """Update the canvas display with current overlay"""
        if self.original_sem is None or self.current_gds is None:
            return
        
        try:
            # Get canvas zoom level
            canvas_zoom = self.canvas_zoom.get_zoom()
            
            # Create overlay with current transparency (no scale_factor here to avoid double zoom)
            overlay_array = self.overlay_composer.compose(
                self.original_sem,
                self.current_gds,
                transparency_percent=self.transform_params['transparency'],
                canvas_size=None,  # Let it use natural size
                scale_factor=1.0   # No scaling in compose, we'll handle it in display
            )
            
            # Apply canvas zoom by resizing the final image
            if canvas_zoom != 1.0:
                h, w = overlay_array.shape[:2]
                new_w = int(w * canvas_zoom)
                new_h = int(h * canvas_zoom)
                overlay_array = cv2.resize(overlay_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Convert to PhotoImage
            photo = create_tkinter_photo(overlay_array, max_size=3000)  # Increase max size
            
            # Clear canvas and display image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Keep reference
            
            # Update scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            messagebox.showerror("Error", f"Display update failed: {str(e)}")
            print(f"Display error details: {e}")
    
    def reset_transforms(self):
        """Reset all transformation parameters"""
        self.move_x_var.set(0)
        self.move_y_var.set(0)
        self.rotation_var.set(0.0)
        self.stretch_x_var.set(1.0)
        self.stretch_y_var.set(1.0)
        self.zoom_var.set(100)
        self.transparency_var.set(40)
        
        self.transform_params = {
            'move_x': 0, 'move_y': 0,
            'rotation': 0.0,
            'stretch_x': 1.0, 'stretch_y': 1.0,
            'zoom': 100,
            'transparency': 40,
            'mirror_horizontal': False,
            'mirror_vertical': False,

        }
        
        if self.original_gds is not None:
            self.current_gds = self.original_gds.copy()
            self.update_display()
    
    def save_transformation(self):
        """Save transformation and create transformed images"""
        if not self.current_file_pair:
            messagebox.showwarning("Warning", "No images loaded")
            return
        
        if self.original_sem is None or self.current_gds is None:
            messagebox.showwarning("Warning", "Need both SEM and GDS images loaded")
            return
        
        try:
            output_name = f"{self.current_file_pair}_transformation"
            
            # 1. Save transformation parameters
            transform_file = os.path.join(self.output_dir, f"{output_name}.json")
            with open(transform_file, 'w') as f:
                json.dump(self.transform_params, f, indent=2)
            
            # 2. Apply inverse transformation to SEM (to align with transformed GDS)
            transformed_sem = self.apply_inverse_transform_to_sem()
            sem_output = os.path.join(self.output_dir, f"{self.current_file_pair}_aligned_sem.png")
            cv2.imwrite(sem_output, transformed_sem)
            
            # 3. Create overlay with 30% GDS transparency
            overlay_array = self.overlay_composer.compose(
                transformed_sem, self.current_gds, transparency_percent=30
            )
            
            overlay_output = os.path.join(self.output_dir, f"{self.current_file_pair}_overlay.png")
            cv2.imwrite(overlay_output, cv2.cvtColor(overlay_array, cv2.COLOR_RGB2BGR))
            
            messagebox.showinfo("Success", f"Saved:\n- {output_name}.json\n- {self.current_file_pair}_aligned_sem.png\n- {self.current_file_pair}_overlay.png")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")
            
    def apply_inverse_transform_to_sem(self):
        """Apply inverse transformation to SEM image to align with transformed GDS"""
        if self.original_sem is None:
            return None
        
        try:
            # Start with original SEM
            transformed_sem = self.original_sem.copy()
            
            # Apply INVERSE transformations in REVERSE order
            # 5. Inverse Mirror (same mirror operation to reverse since it's symmetric)
            if self.transform_params['mirror_horizontal'] or self.transform_params['mirror_vertical']:
                transformed_sem = mirror_image(transformed_sem,
                                               horizontal=self.transform_params['mirror_horizontal'],
                                               vertical=self.transform_params['mirror_vertical'])
                
            # 4. Inverse Move (opposite direction)
            if self.transform_params['move_x'] != 0 or self.transform_params['move_y'] != 0:
                transformed_sem = move_image(transformed_sem, 
                                        -self.transform_params['move_x'],  # Negative
                                        -self.transform_params['move_y'])  # Negative
            
            # 3. Inverse Rotate (opposite angle)
            if self.transform_params['rotation'] != 0.0:
                transformed_sem = rotate_image(transformed_sem, -self.transform_params['rotation'])
            
            # 2. Inverse Stretch
            if self.transform_params['stretch_x'] != 1.0 or self.transform_params['stretch_y'] != 1.0:
                # Inverse of stretch is 1/stretch
                inv_stretch_x = 1.0 / self.transform_params['stretch_x']
                inv_stretch_y = 1.0 / self.transform_params['stretch_y']
                transformed_sem = stretch_image(transformed_sem, inv_stretch_x, inv_stretch_y)
            
            # 1. Inverse Zoom
            if self.transform_params['zoom'] != 100:
                # Inverse of zoom percentage
                inv_zoom = 100 * (100.0 / self.transform_params['zoom'])
                transformed_sem = zoom_image(transformed_sem, int(inv_zoom))
            
            return transformed_sem
            
        except Exception as e:
            print(f"Error applying inverse transform to SEM: {e}")
            return self.original_sem.copy()  # Return original if transformation fails

    def is_identity_transform(self):
        """Check if current transformation is essentially identity (no change)"""
        return (abs(self.transform_params['move_x']) < 1 and
                abs(self.transform_params['move_y']) < 1 and
                abs(self.transform_params['rotation']) < 0.1 and
                abs(self.transform_params['stretch_x'] - 1.0) < 0.01 and
                abs(self.transform_params['stretch_y'] - 1.0) < 0.01 and
                abs(self.transform_params['zoom'] - 100) < 1)
    
    def load_transformation(self):
        """Load transformation parameters from file"""
        if not self.current_file_pair:
            messagebox.showwarning("Warning", "No file pair loaded")
            return
        
        try:
            # Try to load simple parameters first
            simple_file = os.path.join(self.output_dir, f"{self.current_file_pair}_params.json")
            transform_file = os.path.join(self.output_dir, f"{self.current_file_pair}_transform.json")
            
            params = None
            
            # Try simple file first
            if os.path.exists(simple_file):
                with open(simple_file, 'r') as f:
                    params = json.load(f)
            # Fall back to detailed transform file
            elif os.path.exists(transform_file):
                with open(transform_file, 'r') as f:
                    data = json.load(f)
                    if 'transformations' in data:
                        # Convert from detailed format
                        t = data['transformations']
                        params = {
                            'move_x': t.get('move_x_pixels', 0),
                            'move_y': t.get('move_y_pixels', 0),
                            'rotation': t.get('rotation_degrees', 0.0),
                            'stretch_x': t.get('stretch_x_factor', 1.0),
                            'stretch_y': t.get('stretch_y_factor', 1.0),
                            'zoom': t.get('zoom_percent', 100),
                            'transparency': t.get('transparency_percent', 70)
                        }
                    else:
                        params = data  # Old format
            else:
                messagebox.showwarning("Warning", "No saved transformation found for this file pair")
                return
            
            if params:
                # Set UI variables
                self.move_x_var.set(params.get('move_x', 0))
                self.move_y_var.set(params.get('move_y', 0))
                self.rotation_var.set(params.get('rotation', 0.0))
                self.stretch_x_var.set(params.get('stretch_x', 1.0))
                self.stretch_y_var.set(params.get('stretch_y', 1.0))
                self.zoom_var.set(params.get('zoom', 100))
                self.transparency_var.set(params.get('transparency', 70))
                
                # Apply transformations
                self.update_transform()
                
                messagebox.showinfo("Success", f"Transformation loaded for {self.current_file_pair}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load transformation: {str(e)}")
            print(f"Load error details: {e}")


def main():
    root = tk.Tk()
    app = ManualAlignmentUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()