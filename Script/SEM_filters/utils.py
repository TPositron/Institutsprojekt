import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
from PIL import Image, ImageTk

def adjust_var(var, delta, callback=None):
    """
    Adjusts a tk.IntVar or tk.DoubleVar by delta and calls a callback.
    
    Args:
        var: tk.IntVar or tk.DoubleVar to adjust
        delta: Amount to add/subtract (int or float)
        callback: Optional function to call after adjustment
    """
    try:
        current_value = var.get()
        
        if isinstance(var, tk.IntVar):
            new_value = int(current_value + delta)
            # Ensure non-negative for most image processing parameters
            new_value = max(0, new_value)
        elif isinstance(var, tk.DoubleVar):
            new_value = round(current_value + delta, 3)
            # Ensure non-negative for most image processing parameters
            new_value = max(0.0, new_value)
        else:
            raise TypeError(f"Unsupported variable type: {type(var)}")
        
        var.set(new_value)
        
        if callback:
            callback()
            
    except Exception as e:
        print(f"Error adjusting variable: {e}")

def create_button_row(parent, variable, steps=None, callback=None):
    """
    Simplified version of button row function with only up/down buttons.
    
    Args:
        parent: Parent tkinter widget
        variable: tk.IntVar or tk.DoubleVar to control
        steps: Dict with 'small' step size (default: 1 for Int, 0.1 for Double)
        callback: Optional function to call after each adjustment
        
    Returns:
        Frame containing the up/down buttons
    """
    # Set default step sizes
    if steps is None:
        if isinstance(variable, tk.IntVar):
            step_size = 1
        elif isinstance(variable, tk.DoubleVar):
            step_size = 0.1
        else:
            step_size = 1
    else:
        step_size = steps.get('small', 1 if isinstance(variable, tk.IntVar) else 0.1)
    
    # Create button frame
    button_frame = ttk.Frame(parent)
    
    # Up button
    up_button = ttk.Button(
        button_frame, 
        text="▲", 
        width=3,
        command=lambda: adjust_var(variable, step_size, callback)
    )
    up_button.pack(side=tk.TOP, pady=(0, 1))
    
    # Down button  
    down_button = ttk.Button(
        button_frame, 
        text="▼", 
        width=3,
        command=lambda: adjust_var(variable, -step_size, callback)
    )
    down_button.pack(side=tk.BOTTOM, pady=(1, 0))
    
    return button_frame

def normalize_image(image):
    """
    Converts float32 or float64 image into 8-bit uint8 format.
    
    Args:
        image: NumPy array (any dtype)
        
    Returns:
        NumPy array with dtype uint8, values in range [0, 255]
    """
    if image is None:
        raise ValueError("Input image is None")
    
    # If already uint8, return as is
    if image.dtype == np.uint8:
        return image
    
    # Handle different input types
    if image.dtype in [np.float32, np.float64]:
        # Check if values are in [0, 1] range (common for normalized images)
        if np.max(image) <= 1.0 and np.min(image) >= 0.0:
            # Scale from [0, 1] to [0, 255]
            normalized = (image * 255).astype(np.uint8)
        else:
            # Use cv2.normalize for arbitrary float ranges
            normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    elif image.dtype in [np.int16, np.int32, np.int64]:
        # Handle signed integer types
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    elif image.dtype in [np.uint16, np.uint32]:
        # Handle unsigned integer types
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        # Fallback: try to convert directly
        try:
            normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        except:
            # Last resort: clip and convert
            clipped = np.clip(image, 0, 255)
            normalized = clipped.astype(np.uint8)
    
    return normalized

def show_image_on_canvas(canvas, image, zoom_factor=1.0):
    """
    Displays a NumPy image on a Tkinter Canvas, scaled by zoom factor.
    
    Args:
        canvas: tkinter.Canvas widget
        image: NumPy array (grayscale or RGB)
        zoom_factor: Float scaling factor (1.0 = original size)
        
    Returns:
        PhotoImage object (keep reference to prevent garbage collection)
    """
    if image is None:
        raise ValueError("Input image is None")
    
    if canvas is None:
        raise ValueError("Canvas is None")
    
    try:
        # Ensure image is uint8
        display_image = normalize_image(image)
        
        # Handle grayscale vs color images
        if len(display_image.shape) == 2:
            # Grayscale image
            height, width = display_image.shape
            pil_image = Image.fromarray(display_image, mode='L')
        elif len(display_image.shape) == 3:
            # Color image - assume BGR format from OpenCV
            height, width, channels = display_image.shape
            if channels == 3:
                # Convert BGR to RGB for PIL
                rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image, mode='RGB')
            elif channels == 4:
                # RGBA image
                rgba_image = cv2.cvtColor(display_image, cv2.COLOR_BGRA2RGBA)
                pil_image = Image.fromarray(rgba_image, mode='RGBA')
            else:
                raise ValueError(f"Unsupported number of channels: {channels}")
        else:
            raise ValueError(f"Unsupported image shape: {display_image.shape}")
        
        # Apply zoom factor
        if zoom_factor != 1.0:
            new_width = int(width * zoom_factor)
            new_height = int(height * zoom_factor)
            
            # Use appropriate resampling method based on zoom
            if zoom_factor > 1.0:
                # Upscaling - use LANCZOS for better quality
                resample = Image.LANCZOS
            else:
                # Downscaling - use ANTIALIAS for better quality
                resample = Image.ANTIALIAS
                
            pil_image = pil_image.resize((new_width, new_height), resample)
        
        # Convert to PhotoImage
        photo_image = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo_image)
        
        # Update scroll region to match image size
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        # Return PhotoImage to maintain reference
        return photo_image
        
    except Exception as e:
        print(f"Error displaying image on canvas: {e}")
        # Display error message on canvas
        canvas.delete("all")
        canvas.create_text(100, 50, text=f"Error: {str(e)}", anchor=tk.W, fill="red")
        return None

def validate_numeric_entry(value_str, var_type):
    """
    Validates numeric entry input for tkinter Entry widgets.
    
    Args:
        value_str: String value from Entry widget
        var_type: Type of variable (tk.IntVar or tk.DoubleVar)
        
    Returns:
        True if valid, False otherwise
    """
    if value_str == "" or value_str == "-":
        return True  # Allow empty or just minus sign during typing
    
    try:
        if var_type == tk.IntVar:
            int(value_str)
        elif var_type == tk.DoubleVar:
            float(value_str)
        else:
            return False
        return True
    except ValueError:
        return False

def create_parameter_entry(parent, label_text, variable, row, column=0, 
                          include_buttons=True, callback=None, steps=None):
    """
    Creates a complete parameter entry with label, entry field, and optional up/down buttons.
    
    Args:
        parent: Parent tkinter widget
        label_text: Text for the label
        variable: tk.IntVar or tk.DoubleVar
        row: Grid row position
        column: Grid column position (default: 0)
        include_buttons: Whether to include up/down buttons
        callback: Optional callback for button presses
        steps: Step sizes for buttons
        
    Returns:
        Tuple of (label, entry, button_frame) widgets
    """
    # Create label
    label = ttk.Label(parent, text=label_text)
    label.grid(row=row, column=column, sticky=tk.W, padx=(0, 5))
    
    # Create entry with validation
    var_type = type(variable)
    validate_cmd = parent.register(lambda val: validate_numeric_entry(val, var_type))
    
    entry = ttk.Entry(
        parent, 
        textvariable=variable, 
        width=10,
        validate='key',
        validatecommand=(validate_cmd, '%P')
    )
    entry.grid(row=row, column=column + 1, padx=(0, 5))
    
    # Create buttons if requested
    button_frame = None
    if include_buttons:
        button_frame = create_button_row(parent, variable, steps, callback)
        button_frame.grid(row=row, column=column + 2)
    
    return label, entry, button_frame

def get_image_stats(image):
    """
    Calculate basic statistics for an image.
    
    Args:
        image: NumPy array
        
    Returns:
        Dictionary with min, max, mean, std, and dtype information
    """
    if image is None:
        return None
    
    try:
        stats = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'mean': float(np.mean(image)),
            'std': float(np.std(image))
        }
        
        # Add range information
        stats['range'] = stats['max'] - stats['min']
        
        # Add unique values count for small images or binary images
        if image.size < 1000000:  # Less than 1M pixels
            unique_values = len(np.unique(image))
            stats['unique_values'] = unique_values
        
        return stats
        
    except Exception as e:
        print(f"Error calculating image stats: {e}")
        return None

def clamp_value(value, min_val=None, max_val=None):
    """
    Clamp a value between min and max bounds.
    
    Args:
        value: Value to clamp
        min_val: Minimum allowed value (None for no minimum)
        max_val: Maximum allowed value (None for no maximum)
        
    Returns:
        Clamped value
    """
    if min_val is not None:
        value = max(value, min_val)
    if max_val is not None:
        value = min(value, max_val)
    return value

# Example usage and testing
if __name__ == "__main__":
    # Test the utility functions
    import tkinter as tk
    from tkinter import ttk
    
    def test_ui():
        root = tk.Tk()
        root.title("Utils Test")
        
        # Test variables
        int_var = tk.IntVar(value=10)
        double_var = tk.DoubleVar(value=1.5)
        
        # Test parameter entries
        frame = ttk.Frame(root, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        def callback():
            print(f"Int: {int_var.get()}, Double: {double_var.get()}")
        
        create_parameter_entry(frame, "Integer Param:", int_var, 0, callback=callback)
        create_parameter_entry(frame, "Double Param:", double_var, 1, callback=callback)
        
        # Test image display
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        canvas = tk.Canvas(frame, width=200, height=200, bg="white")
        canvas.grid(row=2, column=0, columnspan=3, pady=10)
        
        photo = show_image_on_canvas(canvas, test_image, zoom_factor=1.5)
        
        # Keep reference to prevent garbage collection
        canvas.photo = photo
        
        # Test image stats
        stats = get_image_stats(test_image)
        print("Image Stats:", stats)
        
        root.mainloop()
    
    # Run test if executed directly
    test_ui()