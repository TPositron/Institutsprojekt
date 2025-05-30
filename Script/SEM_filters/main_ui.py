import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import os
import sys

# Add the script directory to Python path
script_dir = r"C:\Users\tarik\Desktop\Bildanalyse\Script"
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, "SEM_filters"))

try:
    # Import custom modules
    from load_image import load_sem_image
    from generate_histogram import generate_histogram
    from filter_manager import FilterManager
    from canvas_zoom import CanvasZoomControl
    from canvas_move import CanvasMoveControl
    from buttons import create_button_row
    
    # Import filter functions
    from SEM_filters.filter_canny import apply_canny
    from SEM_filters.filter_fft_highpass import apply_fft_highpass
    from SEM_filters.filter_gabor import apply_gabor
    from SEM_filters.filter_dog import apply_dog
    from SEM_filters.filter_laplacian import apply_laplacian
    from SEM_filters.filter_wavelet import apply_wavelet_edges
    from SEM_filters.filter_total_variation import apply_total_variation
    from SEM_filters.filter_top_hat import apply_top_hat
    from SEM_filters.filter_threshold import apply_threshold
except ImportError as e:
    print(f"Warning: Could not import custom modules: {e}")
    print("Using fallback implementations...")

class SEMImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("SEM Image Processor")
        self.root.geometry("1400x900")
        
        # Image variables
        self.original_image = None
        self.current_image = None
        self.display_image = None
        self.filter_manager = None
        
        # UI variables
        self.show_histogram = tk.BooleanVar(value=False)
        self.selected_filter = tk.StringVar(value="canny")
        
        # Filter parameters
        self.filter_params = {
            'canny': {
                'low_threshold': tk.IntVar(value=50),
                'high_threshold': tk.IntVar(value=150),
                'aperture_size': tk.IntVar(value=3)
            },
            'fft_highpass': {
                'radius': tk.IntVar(value=30)
            },
            'gabor': {
                'frequency': tk.DoubleVar(value=0.1),
                'theta': tk.DoubleVar(value=0.785),  # π/4
                'sigma_x': tk.DoubleVar(value=5.0),
                'sigma_y': tk.DoubleVar(value=5.0)
            },
            'dog': {
                'sigma1': tk.DoubleVar(value=1.5),
                'sigma2': tk.DoubleVar(value=3.0)
            },
            'laplacian': {
                'ksize': tk.IntVar(value=3)
            },
            'wavelet': {
                'wavelet': tk.StringVar(value='db2'),
                'level': tk.IntVar(value=1)
            },
            'total_variation': {
                'weight': tk.DoubleVar(value=0.2)
            },
            'top_hat': {
                'kernel_size': tk.IntVar(value=5)
            },
            'threshold': {
                'threshold_value': tk.IntVar(value=127),
                'method': tk.StringVar(value='binary')
            }
        }
        
        self.setup_ui()
        self.cleanup_pycache()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel for image and results
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
        
    def setup_left_panel(self, parent):
        # File operations
        file_frame = ttk.LabelFrame(parent, text="File Operations", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load SEM Image", 
                  command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Save Processed Image", 
                  command=self.save_image).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Reset to Original", 
                  command=self.reset_image).pack(fill=tk.X, pady=2)
        
        # Filter selection
        filter_frame = ttk.LabelFrame(parent, text="Filter Selection", padding="10")
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        filters = ["canny", "fft_highpass", "gabor", "dog", "laplacian", 
                  "wavelet", "total_variation", "top_hat", "threshold"]
        
        ttk.Label(filter_frame, text="Select Filter:").pack(anchor=tk.W)
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.selected_filter,
                                   values=filters, state="readonly")
        filter_combo.pack(fill=tk.X, pady=2)
        filter_combo.bind('<<ComboboxSelected>>', self.on_filter_changed)
        
        # Parameters frame
        self.params_frame = ttk.LabelFrame(parent, text="Filter Parameters", padding="10")
        self.params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Actions
        actions_frame = ttk.LabelFrame(parent, text="Actions", padding="10")
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(actions_frame, text="Apply Filter", 
                  command=self.apply_filter).pack(fill=tk.X, pady=2)
        
        # Histogram checkbox
        histogram_frame = ttk.LabelFrame(parent, text="Visualization", padding="10")
        histogram_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(histogram_frame, text="Show Histogram", 
                       variable=self.show_histogram,
                       command=self.update_histogram).pack(anchor=tk.W)
        
        # Initialize parameter display
        self.update_parameter_display()
        
    def setup_right_panel(self, parent):
        # Create main container with two equal panels
        main_container = ttk.Frame(parent)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Image display (50% width)
        image_panel = ttk.Frame(main_container)
        image_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Right side - Results panel (50% width, same size as left menu)
        results_panel = ttk.Frame(main_container, width=350)
        results_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        results_panel.pack_propagate(False)
        
        # Zoom control at the top
        zoom_frame = ttk.LabelFrame(image_panel, text="Zoom Control", padding="5")
        zoom_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Update the zoom control binding section
        try:
            self.zoom_control = CanvasZoomControl(zoom_frame, self.on_zoom_change)
            self.zoom_control.pack(fill=tk.X)
            # Don't bind mouse wheel to zoom control if we have canvas movement
            self.zoom_control.bind_keyboard_shortcuts(self.root)
        except:
            # Fallback zoom control
            ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)
            self.zoom_var = tk.DoubleVar(value=1.0)
            ttk.Scale(zoom_frame, from_=0.1, to=3.0, orient=tk.HORIZONTAL, 
                    variable=self.zoom_var, command=lambda e: self.on_zoom_change()).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add view control buttons
        view_frame = ttk.Frame(zoom_frame)
        view_frame.pack(fill=tk.X, pady=5)

        ttk.Button(view_frame, text="Center View", 
                command=lambda: self.canvas_move.center_image() if hasattr(self, 'canvas_move') else None).pack(side=tk.LEFT, padx=2)
        ttk.Button(view_frame, text="Reset View", 
                command=lambda: self.canvas_move.reset_view() if hasattr(self, 'canvas_move') else None).pack(side=tk.LEFT, padx=2)
        
        # Image display area
        image_frame = ttk.LabelFrame(image_panel, text="Image Display", padding="10")
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbars
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_canvas = tk.Canvas(canvas_frame, bg="white")
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        
        # Add canvas movement control
        self.canvas_move = CanvasMoveControl(self.image_canvas)
        
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind mouse wheel to canvas after creation
        try:
            if hasattr(self, 'zoom_control'):
                self.zoom_control.bind_mouse_wheel(self.image_canvas)
        except:
            pass
        
        # Results panel setup
        # Histogram display
        hist_frame = ttk.LabelFrame(results_panel, text="Histogram", padding="5")
        hist_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.hist_figure = plt.Figure(figsize=(5, 3), dpi=80)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_figure, hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Kernel visualization
        kernel_frame = ttk.LabelFrame(results_panel, text="Filter Kernel", padding="5")
        kernel_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.kernel_canvas = tk.Canvas(kernel_frame, bg="white", height=200)
        self.kernel_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Image statistics display
        stats_frame = ttk.LabelFrame(results_panel, text="Image Statistics", padding="5")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=8, width=30, wrap=tk.WORD)
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)   
    
    def create_parameter_input(self, parent, param_name, param_var, row):
        """Create parameter input with up/down buttons and real-time preview"""
        ttk.Label(parent, text=f"{param_name}:").grid(row=row, column=0, sticky=tk.W, padx=(0, 5))
        
        entry = ttk.Entry(parent, textvariable=param_var, width=10)
        entry.grid(row=row, column=1, padx=(0, 5))
        
        # Bind parameter changes to real-time preview
        param_var.trace_add('write', lambda *args: self.on_parameter_change())
        
        # Up/down buttons
        try:
            button_frame = create_button_row(parent, param_var)
            button_frame.grid(row=row, column=2)
        except:
            # Fallback buttons
            btn_frame = ttk.Frame(parent)
            btn_frame.grid(row=row, column=2)
            
            def increment():
                if isinstance(param_var, tk.IntVar):
                    param_var.set(param_var.get() + 1)
                else:
                    param_var.set(round(param_var.get() + 0.1, 2))
                    
            def decrement():
                if isinstance(param_var, tk.IntVar):
                    param_var.set(max(0, param_var.get() - 1))
                else:
                    param_var.set(round(max(0, param_var.get() - 0.1), 2))
            
            ttk.Button(btn_frame, text="▲", width=3, command=increment).pack(side=tk.TOP)
            ttk.Button(btn_frame, text="▼", width=3, command=decrement).pack(side=tk.BOTTOM)      
    
    def update_parameter_display(self):
        """Update parameter inputs based on selected filter"""
        # Clear existing parameters
        for widget in self.params_frame.winfo_children():
            widget.destroy()
            
        filter_name = self.selected_filter.get()
        if filter_name not in self.filter_params:
            return
            
        params = self.filter_params[filter_name]
        row = 0
        
        for param_name, param_var in params.items():
            if isinstance(param_var, tk.StringVar):
                # Combobox for string parameters
                ttk.Label(self.params_frame, text=f"{param_name}:").grid(row=row, column=0, sticky=tk.W)
                if param_name == 'method':
                    values = ['binary', 'binary_inv', 'tozero', 'tozero_inv', 'trunc']
                elif param_name == 'wavelet':
                    values = ['db2', 'db4', 'haar', 'bior2.2']
                else:
                    values = []
                combo = ttk.Combobox(self.params_frame, textvariable=param_var, values=values, state="readonly")
                combo.grid(row=row, column=1, columnspan=2, sticky=tk.W)
                param_var.trace_add('write', lambda *args: self.on_parameter_change())
            else:
                # Numeric parameters with up/down buttons
                self.create_parameter_input(self.params_frame, param_name, param_var, row)
            row += 1
    
    def on_parameter_change(self):
        """Handle real-time parameter changes"""
        if self.original_image is None:
            return
            
        # Debounce rapid changes
        if hasattr(self, '_parameter_change_job'):
            self.root.after_cancel(self._parameter_change_job)
            
        self._parameter_change_job = self.root.after(300, self._apply_preview_filter)
        
    def _apply_preview_filter(self):
        """Apply filter with current parameters for preview"""
        try:
            filter_name = self.selected_filter.get()
            params = {}
            
            # Get parameters
            for param_name, param_var in self.filter_params[filter_name].items():
                params[param_name] = param_var.get()
            
            # Apply filter to original image for preview
            filtered_image = self.apply_filter_by_name(filter_name, self.original_image, **params)
            
            if filtered_image is not None:
                self.current_image = filtered_image
                self.display_image_on_canvas()
                self.update_histogram()
                self.update_kernel_display()
                
        except Exception as e:
            print(f"Preview filter error: {e}")      
    
    def on_filter_changed(self, event=None):
        """Handle filter selection change"""
        self.update_parameter_display()
        self.update_kernel_display()
        
    def load_image(self):
        """Load SEM image"""
        file_path = filedialog.askopenfilename(
            title="Select SEM Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tiff *.bmp *.tif"),
                    ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Try custom loader first
                try:
                    self.original_image = load_sem_image(file_path)
                except:
                    # Fallback to cv2
                    self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                
                if self.original_image is None:
                    raise ValueError("Failed to load image")
                    
                self.current_image = self.original_image.copy()
                self.filter_manager = FilterManager(self.original_image)
                
                # Clear any existing kernel
                if hasattr(self, 'current_kernel'):
                    delattr(self, 'current_kernel')
                
                self.display_image_on_canvas()
                self.update_histogram()  # This will also update stats
                self.update_kernel_display()
                
                messagebox.showinfo("Success", "Image loaded successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def apply_filter(self):
        """Apply selected filter with current parameters"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        try:
            filter_name = self.selected_filter.get()
            params = {}
            
            # Get parameters
            for param_name, param_var in self.filter_params[filter_name].items():
                params[param_name] = param_var.get()
            
            # Apply filter
            result = self.apply_filter_by_name(filter_name, self.current_image, **params)
            
            if isinstance(result, tuple):
                # Handle filters that return (filtered_image, kernel)
                filtered_image, kernel = result
                self.current_kernel = kernel
            else:
                filtered_image = result
                # Clear kernel for filters that don't produce one
                if hasattr(self, 'current_kernel'):
                    delattr(self, 'current_kernel')
            
            if filtered_image is not None:
                self.current_image = filtered_image
                self.display_image_on_canvas()
                
                # Always update histogram and kernel display after applying filter
                self.update_histogram()
                self.update_kernel_display()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filter: {str(e)}")
            print(f"Filter application error: {e}")  # Debug print            
    
    def display_image_on_canvas(self):
        """Display image on canvas with zoom"""
        if self.current_image is None:
            return
            
        try:
            zoom = getattr(self, 'zoom_control', None)
            zoom_level = zoom.get_zoom() if zoom else getattr(self, 'zoom_var', tk.DoubleVar(value=1.0)).get()
        except:
            zoom_level = 1.0
            
        # Resize image based on zoom
        height, width = self.current_image.shape
        new_width = int(width * zoom_level)
        new_height = int(height * zoom_level)
        
        resized = cv2.resize(self.current_image, (new_width, new_height))
        
        # Convert to PIL and then to PhotoImage
        pil_image = Image.fromarray(resized)
        self.display_image = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
        
        # Update scroll region
        self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
        
    def on_zoom_change(self):
        """Handle zoom change"""
        self.display_image_on_canvas()
        
    def apply_filter(self):
        """Apply selected filter with current parameters"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        try:
            filter_name = self.selected_filter.get()
            params = {}
            
            # Get parameters
            for param_name, param_var in self.filter_params[filter_name].items():
                params[param_name] = param_var.get()
            
            # Apply filter
            filtered_image = self.apply_filter_by_name(filter_name, self.current_image, **params)
            
            if filtered_image is not None:
                self.current_image = filtered_image
                self.display_image_on_canvas()
                self.update_histogram()
                self.update_kernel_display()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filter: {str(e)}")
            
    def apply_filter_by_name(self, filter_name, image, **kwargs):
        """Apply filter by name with parameters"""
        try:
            if filter_name == "canny":
                return apply_canny(image, kwargs['low_threshold'], kwargs['high_threshold'], kwargs['aperture_size'])
            elif filter_name == "fft_highpass":
                return apply_fft_highpass(image, kwargs['radius'])
            elif filter_name == "gabor":
                try:
                    # Fix kernel size calculation
                    ksize = max(int(kwargs['sigma_x'] * 6), int(kwargs['sigma_y'] * 6))
                    if ksize % 2 == 0:
                        ksize += 1  # Ensure odd size
                    ksize = max(ksize, 3)  # Minimum size
                    
                    kernel = cv2.getGaborKernel(
                        (ksize, ksize), kwargs['sigma_x'], kwargs['theta'], 
                        2*np.pi/kwargs['frequency'], kwargs['sigma_y'], 0, ktype=cv2.CV_32F
                    )
                    filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                    self.current_kernel = kernel
                    return filtered
                except Exception as e:
                    print(f"Gabor filter error: {e}")
                    return image
            elif filter_name == "dog":
                return apply_dog(image, kwargs['sigma1'], kwargs['sigma2'])
            elif filter_name == "laplacian":
                return apply_laplacian(image, kwargs['ksize'])
            elif filter_name == "wavelet":
                return apply_wavelet_edges(image, kwargs['wavelet'], kwargs['level'])
            elif filter_name == "total_variation":
                return apply_total_variation(image, kwargs['weight'])
            elif filter_name == "top_hat":
                return apply_top_hat(image, kwargs['kernel_size'])
            elif filter_name == "threshold":
                return apply_threshold(image, kwargs['threshold_value'], kwargs['method'])
                
        except Exception as e:
            # Fallback implementations
            print(f"Using fallback for {filter_name}: {e}")
            return self.fallback_filter(filter_name, image, **kwargs)
            
    def fallback_filter(self, filter_name, image, **kwargs):
        """Fallback filter implementations"""
        if filter_name == "canny":
            return cv2.Canny(image, kwargs['low_threshold'], kwargs['high_threshold'])
        elif filter_name == "threshold":
            _, thresh = cv2.threshold(image, kwargs['threshold_value'], 255, cv2.THRESH_BINARY)
            return thresh
        elif filter_name == "laplacian":
            laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=kwargs['ksize'])
            return cv2.convertScaleAbs(laplacian)
        else:
            return image
    
    def update_histogram(self):
        """Update histogram display and image statistics"""
        if self.current_image is None:
            self.hist_figure.clear()
            self.hist_canvas.draw()
            self.update_image_stats()
            return
            
        try:
            # Create histogram plot
            self.hist_figure.clear()
            ax = self.hist_figure.add_subplot(111)
            
            hist = cv2.calcHist([self.current_image], [0], None, [256], [0, 256])
            ax.plot(hist.flatten(), color='blue', linewidth=1)
            ax.set_title("Image Histogram", fontsize=10)
            ax.set_xlabel("Pixel Value", fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            # Adjust layout
            self.hist_figure.tight_layout()
            self.hist_canvas.draw()
            
            # Update image statistics
            self.update_image_stats()
            
        except Exception as e:
            print(f"Error updating histogram: {e}")
            # Clear histogram on error
            self.hist_figure.clear()
            self.hist_canvas.draw()

    def update_image_stats(self):
        """Update image statistics display"""
        if not hasattr(self, 'stats_text'):
            return
            
        self.stats_text.delete(1.0, tk.END)
        
        if self.current_image is None:
            self.stats_text.insert(tk.END, "No image loaded")
            return
        
        try:
            stats = {
                'Shape': f"{self.current_image.shape}",
                'Data type': str(self.current_image.dtype),
                'Min value': f"{np.min(self.current_image):.2f}",
                'Max value': f"{np.max(self.current_image):.2f}",
                'Mean': f"{np.mean(self.current_image):.2f}",
                'Std dev': f"{np.std(self.current_image):.2f}",
                'Range': f"{np.max(self.current_image) - np.min(self.current_image):.2f}"
            }
            
            # Format statistics text
            stats_text = "Image Statistics:\n" + "-" * 20 + "\n"
            for key, value in stats.items():
                stats_text += f"{key}: {value}\n"
                
            self.stats_text.insert(tk.END, stats_text)
            
        except Exception as e:
            self.stats_text.insert(tk.END, f"Error calculating stats: {e}")
            
    def update_kernel_display(self):
        """Update kernel visualization"""
        if not hasattr(self, 'kernel_canvas'):
            return
            
        self.kernel_canvas.delete("all")
        
        # Check if we have a kernel to display
        if hasattr(self, 'current_kernel') and self.current_kernel is not None:
            try:
                kernel = self.current_kernel
                
                # Handle different kernel types
                if isinstance(kernel, tuple):
                    kernel = kernel[0] if len(kernel) > 0 else None
                    
                if kernel is None or kernel.size == 0:
                    self.kernel_canvas.create_text(100, 100, text="No kernel to display", anchor=tk.CENTER)
                    return
                
                # Normalize kernel for display
                if kernel.dtype != np.uint8:
                    kernel_normalized = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX)
                    kernel_uint8 = kernel_normalized.astype(np.uint8)
                else:
                    kernel_uint8 = kernel.copy()
                
                # Resize kernel for display (make it larger if too small)
                canvas_width = self.kernel_canvas.winfo_width() or 200
                canvas_height = self.kernel_canvas.winfo_height() or 200
                
                if kernel_uint8.shape[0] < 50 or kernel_uint8.shape[1] < 50:
                    # Upscale small kernels
                    scale_factor = min(150 // max(kernel_uint8.shape), 10)
                    new_size = (kernel_uint8.shape[1] * scale_factor, kernel_uint8.shape[0] * scale_factor)
                    kernel_resized = cv2.resize(kernel_uint8, new_size, interpolation=cv2.INTER_NEAREST)
                else:
                    # Downscale large kernels
                    max_size = min(canvas_width - 20, canvas_height - 20, 200)
                    if max(kernel_uint8.shape) > max_size:
                        scale = max_size / max(kernel_uint8.shape)
                        new_size = (int(kernel_uint8.shape[1] * scale), int(kernel_uint8.shape[0] * scale))
                        kernel_resized = cv2.resize(kernel_uint8, new_size)
                    else:
                        kernel_resized = kernel_uint8
                
                # Convert to PIL and display
                pil_kernel = Image.fromarray(kernel_resized)
                self.kernel_image = ImageTk.PhotoImage(pil_kernel)
                
                # Center the image on canvas
                canvas_center_x = canvas_width // 2
                canvas_center_y = canvas_height // 2
                
                self.kernel_canvas.create_image(
                    canvas_center_x, canvas_center_y, 
                    image=self.kernel_image, anchor=tk.CENTER
                )
                
                # Add kernel size info
                info_text = f"Kernel: {kernel.shape}"
                self.kernel_canvas.create_text(
                    10, 10, text=info_text, anchor=tk.NW, 
                    fill="red", font=("Arial", 8, "bold")
                )
                
            except Exception as e:
                print(f"Error displaying kernel: {e}")
                self.kernel_canvas.create_text(
                    100, 100, 
                    text=f"Kernel display error:\n{str(e)}", 
                    anchor=tk.CENTER, fill="red"
                )
        else:
            # No kernel available
            filter_name = self.selected_filter.get()
            self.kernel_canvas.create_text(
                100, 100, 
                text=f"No kernel for\n{filter_name} filter", 
                anchor=tk.CENTER, fill="gray"
            ) 
    
    def save_image(self):
        """Save processed image to specific directory"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image to save!")
            return
        
        # Set default save directory
        save_directory = r"C:\Users\tarik\Desktop\Bildanalyse\Results\SEM_filters"
        
        # Create directory if it doesn't exist
        try:
            os.makedirs(save_directory, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Could not create save directory: {e}")
            save_directory = ""  # Fallback to current directory
        
        # Generate default filename with current filter
        filter_name = self.selected_filter.get()
        default_filename = f"processed_{filter_name}_{len(os.listdir(save_directory)) if os.path.exists(save_directory) else 0:03d}.png"
        
        # Set initial directory and filename
        initialdir = save_directory if os.path.exists(save_directory) else os.getcwd()
        initialfile = default_filename
        
        file_path = filedialog.asksaveasfilename(
            title="Save Processed Image",
            initialdir=initialdir,
            initialfile=initialfile,
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"), 
                ("JPEG files", "*.jpg"), 
                ("TIFF files", "*.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Ensure image is in correct format for saving
                if len(self.current_image.shape) == 2:
                    # Grayscale image
                    cv2.imwrite(file_path, self.current_image)
                else:
                    # Color image - convert BGR to RGB if needed
                    if self.current_image.shape[2] == 3:
                        # Assume BGR format, convert to RGB for saving
                        rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(file_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(file_path, self.current_image)
                
                messagebox.showinfo("Success", f"Image saved successfully to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
                        
    def reset_image(self):
        """Reset to original image"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            if hasattr(self, 'filter_manager'):
                self.filter_manager.reset()
            self.display_image_on_canvas()
            self.update_histogram()
            messagebox.showinfo("Success", "Image reset to original!")
    
    def cleanup_pycache(self):
        """Remove __pycache__ directories"""
        import shutil
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        for root, dirs, files in os.walk(script_dir):
            if '__pycache__' in dirs:
                pycache_path = os.path.join(root, '__pycache__')
                try:
                    shutil.rmtree(pycache_path)
                    print(f"Removed {pycache_path}")
                except Exception as e:
                    print(f"Could not remove {pycache_path}: {e}")

# Fallback FilterManager if not available
class FilterManager:
    def __init__(self, original_image):
        self.original = original_image.copy()
        self.current = original_image.copy()
        self.filter_stack = []

    def reset(self):
        self.current = self.original.copy()
        self.filter_stack = []
        return self.current

if __name__ == "__main__":
    root = tk.Tk()
    app = SEMImageProcessor(root)
    root.mainloop()