import cv2
import matplotlib.pyplot as plt
import numpy as np


def _as_uint8_rgb(img):
    """Convert an image array to uint8 RGB for OpenCV processing."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        scale = 255.0 if np.nanmax(img) <= 1.0 else 1.0
        img = np.clip(img * scale, 0, 255)
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 257.0
    else:
        img = img.astype(np.float32)
        img -= np.min(img)
        max_val = np.max(img)
        if max_val > 0:
            img = img / max_val * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def hist_eq(img):
    img = _as_uint8_rgb(img)
    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # Split the channels
    l, a, b = cv2.split(lab)
    # Apply histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    # Merge the channels
    lab = cv2.merge((l, a, b))
    # Convert back to RGB
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img

def calcExG(img, normalize=False, debug=False, thr = 0.1):
    
    # Normalize image histogram
    img = hist_eq(img)

    # Convert to float
    img = img.astype(np.float32)

    # Calculate ExG
    rgb_sum = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    r = img[:, :, 0] 
    g = img[:, :, 1] 
    b = img[:, :, 2] 
    ExG = (2*g - r - b) / (rgb_sum+0.0001)
    if debug:
        plt.imshow(ExG)
        # Colorbar
        plt.colorbar()
        plt.show()
    
    # Calculate the threshold
    threshold = np.mean(ExG) + thr*np.std(ExG)
    # Threshold the image
    mask = ExG > threshold
    ExG = ExG * mask
    

    # Normalize image   
    if normalize:
        ExG = (ExG - np.min(ExG)) / (np.max(ExG) - np.min(ExG))
        ExG = (ExG * 255).astype(np.uint8)
    if debug:
        plt.imshow(ExG)
        plt.show()

    return ExG

def process_leaf_image(leaf_img, normalize=True, debug=False, sqaure_crop=False, thr=0.1):
    # Calc ExG
    ExG = calcExG(leaf_img, normalize=normalize, debug=debug, thr=thr)
    # Convert to CV_8UC1
    ExG = ExG.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(ExG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours, _ = cv2.findContours(ExG, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour
    leaf_contour = max(contours, key=cv2.contourArea)
    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(leaf_contour)

    if 0:
        # Draw the bounding rectangle
        cv2.rectangle(leaf_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(leaf_img, cv2.COLOR_BGR2RGB))
        plt.show()

    # Calc leaf area in pixels / total area
    leaf_area = np.sum(ExG > 0) / (w*h)
    # print(leaf_area)
    if 0:
        plant_width = w
        plant_height = h
    else:
        # Normalize with image scale, to have a physical dimension lenght
        # Temporalrliy devide by 1200
        scale_factor = 1200
        plant_width = w / scale_factor
        plant_height = h / scale_factor
    # Crop the leaf image
    if sqaure_crop:
        # Calc Center
        cx = x + w//2
        cy = y + h//2
        # Calc the side length
        side = max(w, h)
        # Calc the new bounding box
        x = cx - side//2
        y = cy - side//2
        w = side
        h = side

        # Adjust the crop to fit the image
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + w > leaf_img.shape[1]:
            w = leaf_img.shape[1] - x
        if y + h > leaf_img.shape[0]:
            h = leaf_img.shape[0] - y
    
    leaf_img = leaf_img[y:y+h, x:x+w]

    if 0:
        plt.imshow(cv2.cvtColor(leaf_img, cv2.COLOR_BGR2RGB))
        plt.show()

    return leaf_area, plant_width, plant_height, leaf_img, (x, y, w, h)