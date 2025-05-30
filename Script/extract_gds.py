import gdspy
import numpy as np
import cv2
import os
import math

def export_gds_structure(gds_path, cell_name, layers, bounds, output_path, target_size, rotation_degrees=0):
    """
    Export GDS structure to PNG with optional rotation, maintaining aspect ratio
    """
    # Check if GDS file exists
    if not os.path.exists(gds_path):
        raise FileNotFoundError(f"GDS file not found: {gds_path}")
    
    xmin, ymin, xmax, ymax = bounds
    gds_width = xmax - xmin
    gds_height = ymax - ymin
    
    # Calculate scaling factor to fit within target size while maintaining aspect ratio
    scale = min(target_size[0]/gds_width, target_size[1]/gds_height)
    scaled_width = int(gds_width * scale)
    scaled_height = int(gds_height * scale)
    
    # Create centered image with extra space for rotation if needed
    if abs(rotation_degrees) > 0.1:
        # Create larger canvas to accommodate rotation
        diagonal = int(math.sqrt(target_size[0]**2 + target_size[1]**2))
        temp_size = (diagonal, diagonal)
        image = np.ones(temp_size, dtype=np.uint8) * 255
        temp_offset_x = (diagonal - scaled_width) // 2
        temp_offset_y = (diagonal - scaled_height) // 2
    else:
        image = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * 255
        temp_offset_x = (target_size[0] - scaled_width) // 2
        temp_offset_y = (target_size[1] - scaled_height) // 2
    
    # Load GDS and get polygons
    gds = gdspy.GdsLibrary().read_gds(gds_path)
    cell = gds.top_level()[0] if cell_name == "TOP" else gds.cells[cell_name]
    polygons = cell.get_polygons(by_spec=True)
    
    # Draw all layers
    for layer in layers:
        if (layer, 0) in polygons:
            for poly in polygons[(layer, 0)]:
                if np.any((poly[:, 0] >= xmin) & (poly[:, 0] <= xmax) &
                          (poly[:, 1] >= ymin) & (poly[:, 1] <= ymax)):
                    # Normalize and scale to image space
                    norm_poly = (poly - [xmin, ymin]) * scale
                    int_poly = np.round(norm_poly).astype(np.int32)
                    # Offset to center
                    int_poly += [temp_offset_x, temp_offset_y]
                    cv2.fillPoly(image, [int_poly], color=0)
    
    # Apply rotation if needed
    if abs(rotation_degrees) > 0.1:
        # Rotate the image
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), 
                                borderValue=255, flags=cv2.INTER_LINEAR)
        
        # Crop to target size from center
        center_x, center_y = rotated.shape[1] // 2, rotated.shape[0] // 2
        start_x = center_x - target_size[0] // 2
        start_y = center_y - target_size[1] // 2
        end_x = start_x + target_size[0]
        end_y = start_y + target_size[1]
        
        # Ensure we don't go out of bounds
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(rotated.shape[1], end_x)
        end_y = min(rotated.shape[0], end_y)
        
        final_image = rotated[start_y:end_y, start_x:end_x]
        
        # Pad if necessary to reach target size
        if final_image.shape != (target_size[1], target_size[0]):
            padded = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * 255
            paste_y = (target_size[1] - final_image.shape[0]) // 2
            paste_x = (target_size[0] - final_image.shape[1]) // 2
            padded[paste_y:paste_y+final_image.shape[0], paste_x:paste_x+final_image.shape[1]] = final_image
            final_image = padded
    else:
        final_image = image
    
    cv2.imwrite(output_path, final_image)
    print(f"Saved: {output_path} ({scaled_width}x{scaled_height} content, rotation={rotation_degrees}°, final size={target_size[0]}x{target_size[1]})")

def find_optimal_bounds(polygons, initial_bounds, layers):
    """Find tight bounds around all polygons in specified layers"""
    xmin_init, ymin_init, xmax_init, ymax_init = initial_bounds
    all_points = []
    
    for layer in layers:
        if (layer, 0) in polygons:
            for poly in polygons[(layer, 0)]:
                mask = ((poly[:, 0] >= xmin_init) & (poly[:, 0] <= xmax_init) &
                        (poly[:, 1] >= ymin_init) & (poly[:, 1] <= ymax_init))
                if np.any(mask):
                    all_points.extend(poly[mask])
    
    if not all_points:
        return initial_bounds
    
    all_points = np.array(all_points)
    tight_xmin, tight_ymin = np.min(all_points, axis=0)
    tight_xmax, tight_ymax = np.max(all_points, axis=0)
    
    # Add 5% margin but stay within initial bounds
    margin_x = (tight_xmax - tight_xmin) * 0.05
    margin_y = (tight_ymax - tight_ymin) * 0.05
    
    return (
        max(tight_xmin - margin_x, xmin_init),
        max(tight_ymin - margin_y, ymin_init),
        min(tight_xmax + margin_x, xmax_init),
        min(tight_ymax + margin_y, ymax_init)
    )

def process_structures():
    GDS_PATH = "C:\\Users\\tarik\\Desktop\\Bildanalyse\\Data\\GDS\\Institute_Project_GDS1.gds"
    CELL_NAME = "TOP"
    
    # Create separate output directories
    OUTPUT_1024_DIR = "C:\\Users\\tarik\\Desktop\\Bildanalyse\\Results\\exact_GDS\\1024"
    OUTPUT_666_DIR = "C:\\Users\\tarik\\Desktop\\Bildanalyse\\Results\\exact_GDS\\666"
    OUTPUT_OPTIMIZED_DIR = "C:\\Users\\tarik\\Desktop\\Bildanalyse\\Results\\exact_GDS\\Optimized"
    
    #Create directories if they don't exist
    os.makedirs(OUTPUT_1024_DIR, exist_ok=True)
    os.makedirs(OUTPUT_666_DIR, exist_ok=True)
    os.makedirs(OUTPUT_OPTIMIZED_DIR, exist_ok=True)
    
    structures = {
        1: {'name': 'Circpol_T2', 'initial_bounds': (688.55, 5736.55, 760.55, 5807.1), 'layers': [14]},
        2: {'name': 'IP935Left_11', 'initial_bounds': (693.99, 6406.40, 723.59, 6428.96), 'layers': [1, 2]},
        3: {'name': 'IP935Left_14', 'initial_bounds': (980.959, 6025.959, 1001.770, 6044.979), 'layers': [1]},
        4: {'name': 'QC855GC_CROSS_Bottom', 'initial_bounds': (3730.00, 4700.99, 3756.00, 4760.00), 'layers': [1, 2]},
        5: {'name': 'QC935_46', 'initial_bounds': (7195.558, 5046.99, 7203.99, 5055.33964), 'layers': [1]}
    }
    
    print(f"Reading GDS from: {GDS_PATH}")
    gds = gdspy.GdsLibrary().read_gds(GDS_PATH)
    cell = gds.top_level()[0] if CELL_NAME == "TOP" else gds.cells[CELL_NAME]
    polygons = cell.get_polygons(by_spec=True)
    
    for struct_num, struct_data in structures.items():
        print(f"\nProcessing structure {struct_num} ({struct_data['name']})")
        
        # Version 1: Original 1024x1024
        output_1024 = os.path.join(OUTPUT_1024_DIR, f"structure{struct_num}.png" )
        export_gds_structure(
            GDS_PATH, CELL_NAME, struct_data['layers'],
            struct_data['initial_bounds'], output_1024,
            target_size=(1024, 1024), rotation_degrees=0
        )
        
        # Version 2: Original 1024x666 (cropped height)
        output_666 = os.path.join(OUTPUT_666_DIR, f"structure{struct_num}.png")
        export_gds_structure(
            GDS_PATH, CELL_NAME, struct_data['layers'],
            struct_data['initial_bounds'], output_666,
            target_size=(1024, 666), rotation_degrees=0 
        )
        
        # Version 3: Optimized bounds (centered, minimal background)
        optimal_bounds = find_optimal_bounds(
            polygons, struct_data['initial_bounds'], struct_data['layers'])
        
        output_opt = os.path.join(OUTPUT_OPTIMIZED_DIR, f"structure{struct_num}.png")
        export_gds_structure(
            GDS_PATH, CELL_NAME, struct_data['layers'],
            optimal_bounds, output_opt,
            target_size=(1024, 1024), rotation_degrees=0  # Add this parameter
        )

if __name__ == "__main__":
    process_structures()