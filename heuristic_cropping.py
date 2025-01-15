#!"/Volumes/Public/Work Stuff/heuristic_cropping/heuristic_cropping/.venv/bin/python"
import cv2
import numpy as np


def resize_image(image, max_width=1920, max_height=1080):
    """
    Resizes the image to fit within the specified dimensions while maintaining aspect ratio.
    - max_width: Maximum allowable width of the resized image.
    - max_height: Maximum allowable height of the resized image.
    Increasing these values allows for larger display images, while decreasing them creates smaller outputs.
    """
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def resize_to_height(image, target_height=1024):
    """ 
    Resizes the image to a consistent height while maintaining aspect ratio.
    
    Parameters:
        image (ndarray): Input image to be resized.
        target_height (int): Desired height of the resized image.
        
    Returns:
        resized_image (ndarray): The resized image with consistent height.
        scaling_factor (float): The factor by which the image was resized.
    """
    height, width = image.shape[:2]
    scaling_factor = target_height / height
    new_width = int(width * scaling_factor)
    new_size = (new_width, target_height)  # (width, height)
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image, scaling_factor


def local_stddev(image, ksize=15):
    """
    Computes the local standard deviation over a kernel size in the given image.
    
    Parameters:
        image (ndarray): The input grayscale image.
        ksize (int): Size of the kernel for local computations. Increasing this smoothens the result.
                     For instance, ksize=15 means a 15x15 window is used for computation.
    
    Returns:
        stddev (ndarray): The computed local standard deviation image.
    """
    # Convert image to float32 for precise calculations
    image = image.astype(np.float32)  # Ensures depth is CV_32F

    # Compute mean and squared mean using a box filter
    mean = cv2.boxFilter(image, ddepth=-1, ksize=(ksize, ksize))
    sq_mean = cv2.boxFilter(image**2, ddepth=-1, ksize=(ksize, ksize))

    # Compute standard deviation using the formula: stddev = sqrt(mean_of_square - square_of_mean)
    stddev = cv2.sqrt(sq_mean - mean**2)
    
    return stddev


# Load the image in grayscale
image = cv2.imread(
    r'/Users/kevinmorales/Downloads/trnv images/Northeast Corridor_DTTX-657786.jpg',
    cv2.IMREAD_GRAYSCALE,
)

# Resize the image to a consistent height
resized_image, scaling_factor = resize_to_height(image)

# Apply histogram equalization
equalized_image = cv2.equalizeHist(image)
# Histogram equalization enhances the contrast of the image, making features more distinguishable.
# No adjustable parameters here, but if this step is unnecessary, it can be skipped.

resized_image = resize_image(equalized_image)
cv2.imshow("Equalized", resized_image)
cv2.waitKey(0)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(equalized_image, (1, 1), 0)
# - (5, 5): Kernel size for blurring. Larger kernels smooth the image more but may lose details. Smaller kernels retain details but reduce noise less.
# - 0: Standard deviation for Gaussian kernel. Letting it default (0) automatically calculates it based on kernel size.
cv2.imshow("Blurred", blurred_image)
cv2.waitKey(0)

# Use adaptive thresholding
binary = cv2.adaptiveThreshold(
    blurred_image,
    255,  # Maximum value for thresholded pixels. Higher values make bright regions more distinct.
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive method: Gaussian averages better for smooth gradients.
    cv2.THRESH_BINARY,  # Type of thresholding (binary here).
    11,  # Block size: Size of the local area to calculate the threshold. Smaller sizes detect finer details; larger sizes generalize better.
    2,  # Constant subtracted from mean/weighted mean. Increasing this makes the threshold more sensitive to differences.
)

resized_binary = resize_image(binary)
cv2.imshow("Thresholding", resized_binary)
cv2.waitKey(0)

# Apply morphological operations to clean up noise
kernel = np.ones((3, 3), np.uint8)  # Kernel size for morphological operations. Larger kernels smooth noise more but can merge small regions.
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
# MORPH_CLOSE: Fills small holes inside objects. Useful for connecting broken lines.
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
# MORPH_OPEN: Removes small noise outside objects. Helps refine edges further.

resized_morphology = resize_image(binary)
cv2.imshow("Morphology", resized_morphology)
cv2.waitKey(0)

# Detect vertical edges using Sobel filter
sobel_vertical = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
# - 1, 0: Direction of edges detected (1 for x-direction, 0 for y-direction). Adjust for horizontal (0, 1) if needed.
# - ksize=3: Kernel size for the Sobel operator. Larger sizes detect broader edges but may lose precision.
sobel_vertical = cv2.convertScaleAbs(sobel_vertical)  # Converts gradients to an 8-bit image for visualization.

resized_edges = resize_image(sobel_vertical)
cv2.imshow("Edges (Sobel)", resized_edges)
cv2.waitKey(0)

# NEW STEP: Laplacian filtering for texture analysis
laplacian = cv2.Laplacian(equalized_image, cv2.CV_64F, ksize=3)
# Laplacian filter detects overall changes in intensity, highlighting textured regions.
# - ksize=3: Kernel size for Laplacian. Larger sizes smooth more but may miss finer textures.
laplacian_abs = cv2.convertScaleAbs(laplacian)

resized_laplacian = resize_image(laplacian_abs)
cv2.imshow("Laplacian Texture", resized_laplacian)
cv2.waitKey(0)

# Threshold Laplacian to isolate flat regions
_, flat_regions = cv2.threshold(laplacian_abs, 20, 255, cv2.THRESH_BINARY_INV)
# - 20: Threshold value. Lower values include more regions, higher values exclude faint textures.
# - 255: Max value for thresholded pixels.
resized_flat_regions = resize_image(flat_regions)
cv2.imshow("Flat Regions (Laplacian)", resized_flat_regions)
cv2.waitKey(0)

# NEW STEP: Local standard deviation for flatness detection
stddev = local_stddev(equalized_image, ksize=2)
# Larger `ksize` will emphasize broader flat regions but reduce sensitivity to small details.
stddev = cv2.normalize(stddev, None, 0, 255, cv2.NORM_MINMAX)  # Normalize to display as an 8-bit image.

resized_stddev = resize_image(stddev)
cv2.imshow("Flatness (StdDev)", resized_stddev)
cv2.waitKey(0)

# Combine binary and flatness masks to refine contours
combined_mask = cv2.bitwise_and(binary, flat_regions)
# This step retains only the regions that are both flat and match the morphological processing.
resized_combined = resize_image(combined_mask)
cv2.imshow("Combined Mask", resized_combined)
cv2.waitKey(0)

# Detect contours from the combined mask
contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Adjust thresholds using the scaling factor
min_width = int(50 * scaling_factor)
max_width = int(1200 * scaling_factor)
min_height = int(200 * scaling_factor)
max_height = int(1024 * scaling_factor)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    print(f"Contour: x={x}, y={y}, w={w}, h={h}, aspect_ratio={aspect_ratio}")

    # Adjust aspect ratio and size conditions based on specific needs
    if min_width < w < max_width and min_height < h < max_height:
        # - 0.2 < aspect_ratio < 1.4: Accepts both narrow (partial doors) and square (fully open doors) boxes.
        # - 50 < w < 1200: Limits width of detected boxes.
        # - 200 < h < 1024: Limits height of detected boxes.

        cropped = image[y:max_height, x:x + w]
        resized_cropped = resize_image(cropped)

        # Display the cropped region for manual review
        cv2.imshow(f"Cropped Region {x}_{y}", resized_cropped)
        cv2.waitKey(0)

        # Save the cropped region
        filename = f"cropped_door_{x}_{y}.jpg"
        print(f"Writing {filename}")
        cv2.imwrite(filename, cropped)

# Display the final edges for debugging purposes
resized_edges = resize_image(sobel_vertical)
cv2.imshow("Final Edges", resized_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()