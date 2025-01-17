#!"/Users/kevinmorales/Documents/Work Stuff/heuristic_cropping/heuristic_cropping/heuristic_cropping/.venv/bin/python"
import cv2
import os


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


trnv_image_loc = r'C:\Users\c883206\OneDrive - BNSF Railway\RoboRailCop\2025-01-16_all_trnv_images\trnvopendoor'

for trnv_image in os.listdir(trnv_image_loc):
    if trnv_image.endswith('.jpg'):
        # Load the image in grayscale
        image = cv2.imread(os.path.join(trnv_image_loc, trnv_image), cv2.IMREAD_GRAYSCALE)

        # Step 1: Resize the image to a consistent height
        resized_image, scaling_factor = resize_to_height(image)

        # Step 2: Apply histogram equalization to enhance contrast
        equalized_image = cv2.equalizeHist(image)

        # Step 3: Detect vertical edges using the Sobel filter
        sobel_vertical = cv2.Sobel(equalized_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_vertical_abs = cv2.convertScaleAbs(sobel_vertical)

        resized_edges = resize_image(sobel_vertical_abs)

        # Step 4: Threshold the vertical edges to isolate strong edges
        # 22 for production
        _, binary_edges = cv2.threshold(sobel_vertical_abs, 23, 255, cv2.THRESH_BINARY)
        # - 50: Threshold value. Increase for stricter filtering of edges.

        resized_binary_edges = resize_image(binary_edges)

        # Step 5: Find contours of the vertical edges
        contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 6: Identify container ends
        container_ends = []
        cropped_regions = []
        for (x, y, w, h) in container_ends:
            # Crop a fixed width (e.g., 2000 pixels) from the detected edge
            crop_width = int(2000 * scaling_factor)  # Total crop width
            half_crop_width = crop_width // 2  # Half of the crop width for centering

            # Calculate start_x and end_x
            start_x = x - half_crop_width
            end_x = x + half_crop_width

            # Adjust the bounds to ensure the crop is within the image
            if start_x < 0:  # If the crop goes out of bounds on the left
                start_x = 0
                end_x = crop_width  # Ensure the crop width remains consistent
            elif end_x > image.shape[1]:  # If the crop goes out of bounds on the right
                end_x = image.shape[1]
                start_x = max(0, end_x - crop_width)  # Shift start_x to maintain crop width

            # Crop the region
            cropped_region = image[:, start_x:end_x]
            cropped_regions.append(cropped_region)

            # Resize the cropped region to a consistent height
            resized_cropped_region, _ = resize_to_height(cropped_region)

            # Save the cropped region
            filename = f"cropped_region_{x}_{y}.jpg"
            cv2.imwrite(
                os.path.join(
                    r"C:\Users\c883206\OneDrive - BNSF Railway\RoboRailCop\2025-01-16_all_trnv_images\crops",
                    filename,
                ),
                resized_cropped_region,
            )
            print(f"Saved cropped region: {filename}")

            # Optionally display the cropped region for review
            # resized_cropped = resize_image(cropped_region)
            # cv2.imshow(f"Cropped Region {x}_{y}", resized_cropped)
            # cv2.waitKey(0)

        cv2.destroyAllWindows()