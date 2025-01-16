#!"/Users/kevinmorales/Documents/Work Stuff/heuristic_cropping/heuristic_cropping/heuristic_cropping/.venv/bin/python"
import cv2
import numpy as np


def resize_image(image, max_width=1920*2, max_height=1080*2):
    """Resizes the image to fit within the specified dimensions while maintaining aspect ratio."""
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def resize_to_height(image, target_height=1024):
    """Resizes the image to a consistent height while maintaining aspect ratio."""
    height, width = image.shape[:2]
    scaling_factor = target_height / height
    new_width = int(width * scaling_factor)
    new_size = (new_width, target_height)  # (width, height)
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image, scaling_factor


# Load the image in grayscale
image = cv2.imread(
    r'/Users/kevinmorales/Downloads/trnv images/Southeast Corridor_DTTX-729733.jpg',
    cv2.IMREAD_GRAYSCALE,
)

# Step 1: Resize the image to a consistent height
resized_image, scaling_factor = resize_to_height(image)

# Step 2: Apply histogram equalization to enhance contrast
equalized_image = cv2.equalizeHist(image)

# Step 3: Detect vertical edges using the Sobel filter
sobel_vertical = cv2.Sobel(equalized_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_vertical_abs = cv2.convertScaleAbs(sobel_vertical)

resized_edges = resize_image(sobel_vertical_abs)

# Step 4: Threshold the vertical edges to isolate strong edges
_, binary_edges = cv2.threshold(sobel_vertical_abs, 23, 255, cv2.THRESH_BINARY)
# - 50: Threshold value. Increase for stricter filtering of edges.

resized_binary_edges = resize_image(binary_edges)

# Step 5: Find contours of the vertical edges
contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Identify container ends
container_ends = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = h / w  # Containers are tall, so height > width.
    if 0.2 < aspect_ratio < 15 and w > 5 and h > 500:  # Adjust as needed
        container_ends.append((x, y, w, h))

# Sort container ends by their x-coordinate (horizontal position)
container_ends = sorted(container_ends, key=lambda end: end[0])

# Step 7: Crop regions of interest based on container ends
cropped_regions = []
for (x, y, w, h) in container_ends:
    # Crop a fixed width (e.g., 400 pixels) from the detected edge
    crop_width = int(1400 * scaling_factor)  # Total crop width
    half_crop_width = crop_width // 2  # Half of the crop width for centering

    # Ensure we don't go out of bounds
    start_x = max(0, x - half_crop_width)  # Start at x minus half the crop width
    end_x = min(image.shape[1], x + half_crop_width)  # End at x plus half the crop width

    cropped_region = image[:, start_x:end_x]
    cropped_regions.append(cropped_region)

    # Save the cropped region
    filename = f"cropped_region_{x}_{y}.jpg"
    cv2.imwrite(filename, cropped_region)
    print(f"Saved cropped region: {filename}")

    # Display the cropped region for review
    resized_cropped = resize_image(cropped_region)
    cv2.imshow(f"Cropped Region {x}_{y}", resized_cropped)
    cv2.waitKey(0)

cv2.destroyAllWindows()