import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image, ImageTk
import tempfile
import math

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
from extract_gds import export_gds_structure

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
        
        #SEM button        
        ttk.Button(btn_frame, text="Select SEM", 
                  command=self.select_sem_file).pack(side=tk.LEFT, padx=(0, 5))
        # GDS Structure Selection
        gds_selection_frame = ttk.Frame(file_frame)
        gds_selection_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(gds_selection_frame, text="Select GDS Structure:").pack(anchor=tk.W)

        # Dropdown for GDS structure selection
        self.gds_structure_var = tk.StringVar(value="")
        gds_options = [
            ("Structure 1 - Circpol_T2", "1"),
            ("Structure 2 - IP935Left_11", "2"), 
            ("Structure 3 - IP935Left_14", "3"),
            ("Structure 4 - QC855GC_CROSS_Bottom", "4"),
            ("Structure 5 - QC935_46", "5")
        ]

        gds_dropdown = ttk.Combobox(gds_selection_frame, textvariable=self.gds_structure_var,
                                values=[option[0] for option in gds_options],
                                state="readonly", width=30)
        gds_dropdown.pack(pady=(5, 0))
        gds_dropdown.bind('<<ComboboxSelected>>', self.on_gds_structure_selected)
        
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
        ttk.Button(action_frame, text="Generate Aligned GDS", 
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
            initialdir=self.sem_cropped_dir,
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tiff *.bmp")]
        )
        if filename:
            self.original_sem = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if self.original_sem is None:
                messagebox.showerror("Error", "Failed to load SEM image")
                return
                
            base_name = os.path.splitext(os.path.basename(filename))[0]
            
            # Update file pair name - preserve GDS structure info if already selected
            if self.current_file_pair and "struct" in self.current_file_pair:
                # Keep the structure info, update SEM part
                struct_part = self.current_file_pair.split("_struct")[-1]
                self.current_file_pair = f"{base_name}_struct{struct_part}"
            else:
                self.current_file_pair = base_name
                
            messagebox.showinfo("Success", f"Loaded SEM: {base_name}")
            self.update_display()
    
    def on_gds_structure_selected(self, event=None):
        """Handle GDS structure selection from dropdown"""
        selected_text = self.gds_structure_var.get()
        if not selected_text:
            return
        
        # Extract structure number from selection
        structure_num = selected_text.split()[1]  # Gets "1", "2", etc.
        # Store the current structure number for later use
        self.current_structure_num = int(structure_num)
        
        try:
            # Generate GDS image using extract_gds functions
            print(f"Generating GDS structure {structure_num}...")
            self.original_gds = self.generate_gds_preview(int(structure_num))
            
            if self.original_gds is None:
                messagebox.showerror("Error", "Failed to generate GDS preview")
                return
                
            self.current_gds = self.original_gds.copy()
            
            # Update current_file_pair to include structure info
            if self.current_file_pair:
                # If SEM is already loaded, append structure info
                self.current_file_pair = f"{self.current_file_pair}_struct{structure_num}"
            else:
                # If no SEM loaded yet, just use structure info
                self.current_file_pair = f"structure{structure_num}"
            
            messagebox.showinfo("Success", f"Generated GDS Structure {structure_num}")
            self.reset_transforms()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate GDS structure: {str(e)}")
            print(f"Detailed error: {e}")
    
    def generate_gds_preview(self, structure_num):
        """Generate GDS preview image for the selected structure"""
        try:
            # GDS file path - update this path as needed
            GDS_PATH = "C:\\Users\\tarik\\Desktop\\Bildanalyse\\Data\\GDS\\Institute_Project_GDS1.gds"
            CELL_NAME = "TOP"
            
            # Structure definitions (matching extract_gds.py)
            structures = {
                1: {'name': 'Circpol_T2', 'initial_bounds': (688.55, 5736.55, 760.55, 5807.1), 'layers': [14]},
                2: {'name': 'IP935Left_11', 'initial_bounds': (693.99, 6406.40, 723.59, 6428.96), 'layers': [1, 2]},
                3: {'name': 'IP935Left_14', 'initial_bounds': (980.959, 6025.959, 1001.770, 6044.979), 'layers': [1]},
                4: {'name': 'QC855GC_CROSS_Bottom', 'initial_bounds': (3730.00, 4700.99, 3756.00, 4760.00), 'layers': [1, 2]},
                5: {'name': 'QC935_46', 'initial_bounds': (7195.558, 5046.99, 7203.99, 5055.33964), 'layers': [1]}
            }
            
            if structure_num not in structures:
                raise ValueError(f"Structure {structure_num} not found")
            
            struct_data = structures[structure_num]
            
            # Create temporary file for the preview
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Generate the GDS image using extract_gds function
                export_gds_structure(
                    GDS_PATH, CELL_NAME, struct_data['layers'],
                    struct_data['initial_bounds'], temp_path,
                    target_size=(1024, 1024)
                )
                
                # Load the generated image
                gds_image = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
                
                if gds_image is None:
                    raise ValueError("Failed to load generated GDS image")
                    
                return gds_image
                
            finally:
                # Always clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
        except Exception as e:
            print(f"Error generating GDS preview: {e}")
            return None
    
    def update_transform(self):
        """Update transformation parameters and apply to GDS image"""
        if self.original_gds is None:
            return
        
        try:
            # Get current parameter values
            # Get current parameter values with validation
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

            # Validate parameters
            validation_errors = self.validate_transformation_params()
            if validation_errors:
                print("Transformation validation warnings:")
                for error in validation_errors:
                    print(f"  - {error}")
                        
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
        """Save transformation and generate aligned GDS image"""
        if not hasattr(self, 'current_structure_num') or not self.current_structure_num:
            messagebox.showwarning("Warning", "No GDS structure selected")
            return
        
        if self.original_sem is None or self.current_gds is None:
            messagebox.showwarning("Warning", "Need both SEM and GDS images loaded")
            return
        
        try:
            # Get structure name for filename
            structures = {
                1: 'Circpol_T2',
                2: 'IP935Left_11', 
                3: 'IP935Left_14',
                4: 'QC855GC_CROSS_Bottom',
                5: 'QC935_46'
            }
            
            structure_name = structures.get(self.current_structure_num, f"structure{self.current_structure_num}")
            
            # 1. Save transformation parameters
            transform_file = os.path.join(self.output_dir, f"{structure_name}_transformation.json")
            with open(transform_file, 'w') as f:
                json.dump(self.transform_params, f, indent=2)
            
            # 2. Generate and save pixel-aligned GDS image (1024x666)
            aligned_gds_path, new_bounds = self.save_aligned_gds_image(
                self.current_structure_num, 
                target_size=(1024, 666)
            )
            
            if aligned_gds_path:
                success_msg = f"Generated aligned GDS:\n"
                success_msg += f"• {structure_name}_transformation.json\n"
                success_msg += f"• {structure_name}_aligned_gds.png"
                messagebox.showinfo("Success", success_msg)
            else:
                messagebox.showerror("Error", "Failed to generate aligned GDS")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save transformation: {str(e)}")
            print(f"Save error details: {e}")             
    
    def generate_aligned_gds_image(self, structure_num, target_size=(1024, 666)):
        """
        Generate a new GDS image that is pixel-aligned with the SEM image.
        Takes transformation parameters and converts them back to GDS coordinates.
        """
        try:
            # Structure definitions (same as in extract_gds.py)
            structures = {
                1: {'name': 'Circpol_T2', 'initial_bounds': (688.55, 5736.55, 760.55, 5807.1), 'layers': [14]},
                2: {'name': 'IP935Left_11', 'initial_bounds': (693.99, 6406.40, 723.59, 6428.96), 'layers': [1, 2]},
                3: {'name': 'IP935Left_14', 'initial_bounds': (980.959, 6025.959, 1001.770, 6044.979), 'layers': [1]},
                4: {'name': 'QC855GC_CROSS_Bottom', 'initial_bounds': (3730.00, 4700.99, 3756.00, 4760.00), 'layers': [1, 2]},
                5: {'name': 'QC935_46', 'initial_bounds': (7195.558, 5046.99, 7203.99, 5055.33964), 'layers': [1]}
            }
            
            if structure_num not in structures:
                raise ValueError(f"Structure {structure_num} not found")
                
            struct_data = structures[structure_num]
            original_bounds = struct_data['initial_bounds']
            xmin_orig, ymin_orig, xmax_orig, ymax_orig = original_bounds
            
            # Calculate original GDS dimensions and pixel scale for 1024x1024 reference
            gds_width_orig = xmax_orig - xmin_orig
            gds_height_orig = ymax_orig - ymin_orig
            
            # Calculate scale factors based on original 1024x1024 image
            scale_x_orig = gds_width_orig / 1024
            scale_y_orig = gds_height_orig / 1024
            
            # Calculate new bounds based on transformation parameters
            new_bounds = self.calculate_transformed_gds_bounds(
                original_bounds, scale_x_orig, scale_y_orig, target_size
            )
            
            # Generate GDS image with new bounds and rotation
            GDS_PATH = "C:\\Users\\tarik\\Desktop\\Bildanalyse\\Data\\GDS\\Institute_Project_GDS1.gds"
            CELL_NAME = "TOP"
            
            # Create temporary file for the aligned GDS image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Generate the aligned GDS image WITH rotation applied
                export_gds_structure(
                    GDS_PATH, CELL_NAME, struct_data['layers'],
                    new_bounds, temp_path, target_size=target_size,
                    rotation_degrees=self.transform_params['rotation']  # Apply rotation directly
                )
                
                # Load the generated image
                aligned_gds = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
                
                if aligned_gds is None:
                    raise ValueError("Failed to load generated aligned GDS image")
                
                print(f"Generated aligned GDS image:")
                print(f"  Original bounds: {original_bounds}")
                print(f"  New bounds: {new_bounds}")  
                print(f"  Target size: {target_size}")
                print(f"  Applied rotation: {self.transform_params['rotation']}°")
                
                return aligned_gds, new_bounds
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            print(f"Error generating aligned GDS image: {e}")
            return None, None
    
    def calculate_transformed_gds_bounds(self, original_bounds, scale_x, scale_y, target_size):
        """
        Calculate new GDS bounds that account for ALL transformation parameters.
        Applies INVERSE transformations to expand GDS bounds appropriately.
        """
        xmin_orig, ymin_orig, xmax_orig, ymax_orig = original_bounds
        target_width, target_height = target_size
        
        # Original center in GDS coordinates
        center_x_orig = (xmin_orig + xmax_orig) / 2
        center_y_orig = (ymin_orig + ymax_orig) / 2
        
        # Original dimensions in GDS units
        orig_width_gds = xmax_orig - xmin_orig
        orig_height_gds = ymax_orig - ymin_orig
        
        # 1. Calculate inverse transformation factors
        zoom_factor = self.transform_params['zoom'] / 100.0
        zoom_inv = 1.0 / zoom_factor if zoom_factor > 0 else 1.0
        
        stretch_x_inv = 1.0 / self.transform_params['stretch_x'] if self.transform_params['stretch_x'] > 0 else 1.0
        stretch_y_inv = 1.0 / self.transform_params['stretch_y'] if self.transform_params['stretch_y'] > 0 else 1.0
        
        # 2. Calculate effective scaling needed for target size
        # We want the final image to be target_size, so work backwards
        effective_scale_x = (target_width * scale_x * zoom_inv * stretch_x_inv) / orig_width_gds
        effective_scale_y = (target_height * scale_y * zoom_inv * stretch_y_inv) / orig_height_gds
        
        # 3. Calculate new dimensions in GDS units
        new_gds_width = orig_width_gds * effective_scale_x
        new_gds_height = orig_height_gds * effective_scale_y
        
        # 4. Account for pixel movement (convert to GDS coordinates)
        move_x_gds = -self.transform_params['move_x'] * scale_x * zoom_inv
        move_y_gds = -self.transform_params['move_y'] * scale_y * zoom_inv
        
        # 5. Account for rotation by expanding bounds
        rotation_rad = math.radians(abs(self.transform_params['rotation']))
        if abs(self.transform_params['rotation']) > 0.1:
            # Calculate bounding box expansion for rotation
            cos_rot = abs(math.cos(rotation_rad))
            sin_rot = abs(math.sin(rotation_rad))
            
            # Rotated bounding box dimensions
            rotated_width = new_gds_width * cos_rot + new_gds_height * sin_rot
            rotated_height = new_gds_width * sin_rot + new_gds_height * cos_rot
            
            new_gds_width = rotated_width
            new_gds_height = rotated_height
        
        # 6. Calculate new center (offset by inverse movement)
        new_center_x = center_x_orig + move_x_gds
        new_center_y = center_y_orig + move_y_gds
        
        # 7. Calculate final bounds
        new_xmin = new_center_x - new_gds_width / 2
        new_xmax = new_center_x + new_gds_width / 2  
        new_ymin = new_center_y - new_gds_height / 2
        new_ymax = new_center_y + new_gds_height / 2
        
        # 8. Add safety margin (2% of original dimensions)
        margin_x = orig_width_gds * 0.02
        margin_y = orig_height_gds * 0.02
        
        final_bounds = (
            new_xmin - margin_x,
            new_ymin - margin_y, 
            new_xmax + margin_x,
            new_ymax + margin_y
        )
        
        print(f"GDS Bounds Calculation:")
        print(f"  Original: {original_bounds}")
        print(f"  Transform params: move=({self.transform_params['move_x']}, {self.transform_params['move_y']}), "
            f"zoom={self.transform_params['zoom']}%, stretch=({self.transform_params['stretch_x']}, {self.transform_params['stretch_y']}), "
            f"rotation={self.transform_params['rotation']}°")
        print(f"  Final bounds: {final_bounds}")
        
        return final_bounds
    
    def save_aligned_gds_image(self, structure_num, target_size=(1024, 666)):
        """
        Generate and save a pixel-aligned GDS image that matches the SEM alignment.
        """
        try:
            # Structure name mapping
            structure_names = {
                1: 'Circpol_T2',
                2: 'IP935Left_11', 
                3: 'IP935Left_14',
                4: 'QC855GC_CROSS_Bottom',
                5: 'QC935_46'
            }
            
            structure_name = structure_names.get(structure_num, f"structure{structure_num}")
            
            # Generate the aligned GDS image
            aligned_gds, new_bounds = self.generate_aligned_gds_image(structure_num, target_size)
            
            if aligned_gds is None:
                print("Failed to generate aligned GDS image")
                return None, None
            
            # Save the aligned GDS image with structure name
            output_path = os.path.join(self.output_dir, f"{structure_name}_aligned_gds.png")
            cv2.imwrite(output_path, aligned_gds)
            
            print(f"Saved aligned GDS image: {output_path}")
            print(f"New GDS bounds: {new_bounds}")
            
            return output_path, new_bounds
            
        except Exception as e:
            print(f"Failed to save aligned GDS: {str(e)}")
            return None, None
    
    def is_identity_transform(self):
        """Check if current transformation is essentially identity (no change)"""
        return (abs(self.transform_params['move_x']) < 1 and
                abs(self.transform_params['move_y']) < 1 and
                abs(self.transform_params['rotation']) < 0.1 and
                abs(self.transform_params['stretch_x'] - 1.0) < 0.01 and
                abs(self.transform_params['stretch_y'] - 1.0) < 0.01 and
                abs(self.transform_params['zoom'] - 100) < 1)
  
    def validate_transformation_params(self):
        """Validate that transformation parameters are within acceptable ranges"""
        validation_errors = []
        
        # Check rotation bounds
        if not (-90 <= self.transform_params['rotation'] <= 90):
            validation_errors.append(f"Rotation {self.transform_params['rotation']}° is outside valid range (-90° to +90°)")
        
        # Check stretch bounds
        if not (0.1 <= self.transform_params['stretch_x'] <= 3.0):
            validation_errors.append(f"Stretch X {self.transform_params['stretch_x']} is outside valid range (0.1 to 3.0)")
        
        if not (0.1 <= self.transform_params['stretch_y'] <= 3.0):
            validation_errors.append(f"Stretch Y {self.transform_params['stretch_y']} is outside valid range (0.1 to 3.0)")
        
        # Check zoom bounds
        if not (10 <= self.transform_params['zoom'] <= 300):
            validation_errors.append(f"Zoom {self.transform_params['zoom']}% is outside valid range (10% to 300%)")
        
        # Check transparency bounds
        if not (0 <= self.transform_params['transparency'] <= 100):
            validation_errors.append(f"Transparency {self.transform_params['transparency']}% is outside valid range (0% to 100%)")
        
        return validation_errors
    
    def load_transformation(self):
        """Load transformation parameters from file"""
        if not hasattr(self, 'current_structure_num') or not self.current_structure_num:
            messagebox.showwarning("Warning", "No GDS structure selected")
            return
        
        try:
            # Get structure name
            structures = {
                1: 'Circpol_T2',
                2: 'IP935Left_11', 
                3: 'IP935Left_14',
                4: 'QC855GC_CROSS_Bottom',
                5: 'QC935_46'
            }
            
            structure_name = structures.get(self.current_structure_num, f"structure{self.current_structure_num}")
            
            # Try to load transformation file
            transform_file = os.path.join(self.output_dir, f"{structure_name}_transformation.json")
            
            if not os.path.exists(transform_file):
                messagebox.showwarning("Warning", f"No saved transformation found for {structure_name}")
                return
            
            with open(transform_file, 'r') as f:
                params = json.load(f)
            
            # Set UI variables
            self.move_x_var.set(params.get('move_x', 0))
            self.move_y_var.set(params.get('move_y', 0))
            self.rotation_var.set(params.get('rotation', 0.0))
            self.stretch_x_var.set(params.get('stretch_x', 1.0))
            self.stretch_y_var.set(params.get('stretch_y', 1.0))
            self.zoom_var.set(params.get('zoom', 100))
            self.transparency_var.set(params.get('transparency', 40))
            
            # Update transform_params and apply transformations
            self.update_transform()
            
            messagebox.showinfo("Success", f"Transformation loaded for {structure_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load transformation: {str(e)}")
            print(f"Load error details: {e}")

def main():
    root = tk.Tk()
    app = ManualAlignmentUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
    
 