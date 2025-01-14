import cv2
import numpy as np

def resize_image(image, max_width=1920, max_height=1080):
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

image = cv2.imread(r"C:\Users\c883206\OneDrive - BNSF Railway\RoboRailCop\error_images\DTTX-0000786742_L8_20241003_125720_TRNV_83847_Melrose1_B02C2.jpg", cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization
equalized_image = cv2.equalizeHist(image)

resized_image = resize_image(equalized_image)
cv2.imshow("equalized", resized_image)
cv2.waitKey(0)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

# Use adaptive thresholding
binary = cv2.adaptiveThreshold(blurred_image, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

resized_binary = resize_image(binary)
cv2.imshow("thresholding", resized_binary)
cv2.waitKey(0)

# Apply morphological operations
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

resized_morphology = resize_image(binary)
cv2.imshow("morphology", resized_morphology)
cv2.waitKey(0)

edges = cv2.Canny(binary, threshold1=50, threshold2=100)

resized_edges = resize_image(edges)
cv2.imshow("edges", resized_edges)
cv2.waitKey(0)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h

    # Adjust aspect ratio and size conditions
    if 0.8 < aspect_ratio < 1.2 and w > 200 and h > 200:
        cropped = image[y:y+h, x:x+w]
        filename = f"cropped_door_{x}_{y}.jpg"
        print(f"writing {filename}")
        cv2.imwrite(filename, cropped)