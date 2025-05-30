import cv2
import os

def load_sem_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image

if __name__ == "__main__":
    try:
        example_path = "C:\\Users\\tarik\\Desktop\\Bildanalyse\\Results\\Centers\\SEM\\cropped"
        img = load_sem_image(example_path)
        print(f"Loaded image: {img.shape} {img.dtype}")
    except Exception as e:
        print(f"Error: {e}")