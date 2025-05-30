import numpy as np
import cv2
from PIL import Image, ImageTk


def center_image(image, canvas_size):
    """
    Center an image on a canvas of specified size.
    
    Args:
        image: Input image (numpy array, grayscale or RGB)
        canvas_size: Tuple (width, height) for the canvas dimensions
        
    Returns:
        tuple: (centered_image, position) where position is (start_x, start_y)
    """
    canvas_w, canvas_h = canvas_size
    
    if len(image.shape) == 2:  # Grayscale
        img_h, img_w = image.shape
        canvas = np.ones((canvas_h, canvas_w), dtype=image.dtype) * 255
    else:  # RGB
        img_h, img_w = image.shape[:2]
        canvas = np.ones((canvas_h, canvas_w, image.shape[2]), dtype=image.dtype) * 255
    
    # Calculate center position
    start_x = max(0, (canvas_w - img_w) // 2)
    start_y = max(0, (canvas_h - img_h) // 2)
    
    # Calculate bounds to handle images larger than canvas
    end_x = min(start_x + img_w, canvas_w)
    end_y = min(start_y + img_h, canvas_h)
    
    # Calculate crop dimensions if image is larger than canvas
    crop_w = end_x - start_x
    crop_h = end_y - start_y
    
    # Place image on canvas
    if len(image.shape) == 2:  # Grayscale
        canvas[start_y:end_y, start_x:end_x] = image[:crop_h, :crop_w]
    else:  # RGB
        canvas[start_y:end_y, start_x:end_x] = image[:crop_h, :crop_w]
    
    return canvas, (start_x, start_y)


def apply_transparency(gds_image, transparency_percent, background_rgb=None, overlay_color=(255, 100, 150)):
    """
    Apply transparency to a GDS image for overlay display.
    
    Args:
        gds_image: Grayscale GDS image (numpy array)
        transparency_percent: Transparency value from 0 (invisible) to 100 (opaque)
        background_rgb: RGB background image (if None, uses white background)
        overlay_color: RGB tuple for the GDS overlay color
        
    Returns:
        RGB image with transparency applied
    """
    alpha = transparency_percent / 100.0
    h, w = gds_image.shape
    
    # Create or use provided RGB background
    if background_rgb is None:
        result = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)
    else:
        # Ensure background is the same size as GDS image
        if background_rgb.shape[:2] != (h, w):
            background_rgb = cv2.resize(background_rgb, (w, h))
        result = background_rgb.copy()
    
    # Create mask for GDS content (dark pixels indicate structures)
    mask = gds_image < 200
    
    # Apply transparent overlay where GDS structures exist
    for channel in range(3):
        result[:, :, channel][mask] = (
            result[:, :, channel][mask] * (1 - alpha) + 
            overlay_color[channel] * alpha
        ).astype(np.uint8)
    
    return result


def compose_overlay(sem_image, gds_image, transparency_percent=70, 
                   canvas_size=None, overlay_color=(255, 100, 150), 
                   scale_factor=1.0):
    """
    Compose SEM and GDS images into a single overlay display.
    
    Args:
        sem_image: SEM background image (grayscale numpy array)
        gds_image: GDS overlay image (grayscale numpy array)  
        transparency_percent: GDS transparency (0-100)
        canvas_size: Optional tuple (width, height) for canvas size
        overlay_color: RGB tuple for GDS overlay color
        scale_factor: Scale factor for final display size
        
    Returns:
        PIL.Image: RGB image ready for display
    """
    if sem_image is None or gds_image is None:
        raise ValueError("Both SEM and GDS images must be provided")
    
    # Determine canvas size
    if canvas_size is None:
        sem_h, sem_w = sem_image.shape
        gds_h, gds_w = gds_image.shape
        canvas_w = max(sem_w, gds_w, 1024)
        canvas_h = max(sem_h, gds_h, 1024)
    else:
        canvas_w, canvas_h = canvas_size
    
    # Center SEM image on canvas
    sem_centered, sem_pos = center_image(sem_image, (canvas_w, canvas_h))
    
    # Convert SEM to RGB
    sem_rgb = np.stack([sem_centered] * 3, axis=-1)
    
    # Center GDS image on canvas
    gds_centered, gds_pos = center_image(gds_image, (canvas_w, canvas_h))
    
    # Apply transparency to GDS overlay
    gds_transparent = apply_transparency(
        gds_centered, 
        transparency_percent, 
        background_rgb=sem_rgb,
        overlay_color=overlay_color
    )
    
    # Apply scaling if requested
    if scale_factor != 1.0:
        new_w = int(canvas_w * scale_factor)
        new_h = int(canvas_h * scale_factor)
        gds_transparent = cv2.resize(gds_transparent, (new_w, new_h), 
                                   interpolation=cv2.INTER_LINEAR)
    
    return gds_transparent


def create_display_image(overlay_array, max_size=1024):
    """
    Convert numpy array to PIL Image with optional size limiting.
    
    Args:
        overlay_array: RGB numpy array
        max_size: Maximum dimension for display scaling
        
    Returns:
        PIL.Image: Image ready for Tkinter display
    """
    if overlay_array is None:
        raise ValueError("Overlay array cannot be None")
    
    # Ensure array is in correct format
    if len(overlay_array.shape) != 3 or overlay_array.shape[2] != 3:
        raise ValueError("Array must be RGB (height, width, 3)")
    
    # Scale down if too large
    h, w = overlay_array.shape[:2]
    if w > max_size or h > max_size:
        scale = min(max_size / w, max_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        overlay_array = cv2.resize(overlay_array, (new_w, new_h))
    
    # Convert to PIL Image
    return Image.fromarray(overlay_array)


def create_tkinter_photo(overlay_array, max_size=1024):
    """
    Create a Tkinter-compatible PhotoImage from numpy array.
    
    Args:
        overlay_array: RGB numpy array
        max_size: Maximum dimension for display scaling
        
    Returns:
        ImageTk.PhotoImage: Ready for Tkinter canvas display
    """
    pil_image = create_display_image(overlay_array, max_size)
    return ImageTk.PhotoImage(pil_image)


class OverlayComposer:
    """
    Class to handle overlay composition with state management.
    """
    
    def __init__(self, default_transparency=70, default_overlay_color=(255, 100, 150)):
        self.default_transparency = default_transparency
        self.default_overlay_color = default_overlay_color
        self.last_sem = None
        self.last_gds = None
        self.last_result = None
        
    def compose(self, sem_image, gds_image, transparency_percent=None, 
                canvas_size=None, overlay_color=None, scale_factor=1.0):
        """
        Compose overlay with caching for performance.
        
        Args:
            sem_image: SEM background image
            gds_image: GDS overlay image
            transparency_percent: GDS transparency (uses default if None)
            canvas_size: Canvas size tuple
            overlay_color: Overlay color tuple (uses default if None)
            scale_factor: Display scale factor
            
        Returns:
            RGB numpy array of composed overlay
        """
        # Use defaults if not specified
        if transparency_percent is None:
            transparency_percent = self.default_transparency
        if overlay_color is None:
            overlay_color = self.default_overlay_color
            
        # Simple caching - recompose only if images changed
        if (sem_image is not self.last_sem or gds_image is not self.last_gds or 
            self.last_result is None):
            
            self.last_result = compose_overlay(
                sem_image, gds_image, transparency_percent,
                canvas_size, overlay_color, scale_factor
            )
            self.last_sem = sem_image
            self.last_gds = gds_image
            
        return self.last_result
    
    def create_photo(self, sem_image, gds_image, transparency_percent=None,
                    canvas_size=None, overlay_color=None, scale_factor=1.0, 
                    max_size=1500):
        """
        Create Tkinter PhotoImage directly from composition.
        
        Returns:
            ImageTk.PhotoImage ready for display
        """
        overlay = self.compose(sem_image, gds_image, transparency_percent,
                             canvas_size, overlay_color, scale_factor)
        return create_tkinter_photo(overlay, max_size)
    
    def reset_cache(self):
        """Reset internal cache to force recomposition."""
        self.last_sem = None
        self.last_gds = None
        self.last_result = None


# Convenience functions for common use cases
def quick_overlay(sem_image, gds_image, transparency=70):
    """Quick overlay composition with default settings."""
    return compose_overlay(sem_image, gds_image, transparency)


def quick_tkinter_overlay(sem_image, gds_image, transparency=70, max_size=1500):
    """Quick Tkinter PhotoImage creation from overlay."""
    overlay = quick_overlay(sem_image, gds_image, transparency)
    return create_tkinter_photo(overlay, max_size)
