#!"/Users/kevinmorales/Documents/Work Stuff/heuristic_cropping/heuristic_cropping/heuristic_cropping/.venv/bin/python"

import os
import cv2
import shutil

# Define the source directory containing all images
source_dir = r'C:\Users\c883206\OneDrive - BNSF Railway\RoboRailCop\2025-01-16_all_trnv_images\crops'

# Define the target directory where images will be organized by class
closed_door_dir = r"C:\Users\c883206\OneDrive - BNSF Railway\RoboRailCop\2025-01-16_all_trnv_images\crops\closed_door"
container_side_dir = r"C:\Users\c883206\OneDrive - BNSF Railway\RoboRailCop\2025-01-16_all_trnv_images\crops\cont_sides"
open_door_dir = r"C:\Users\c883206\OneDrive - BNSF Railway\RoboRailCop\2025-01-16_all_trnv_images\crops\open_door"
handhold_dir = r"C:\Users\c883206\OneDrive - BNSF Railway\RoboRailCop\2025-01-16_all_trnv_images\crops\handholds"
cont_face_dir = r"C:\Users\c883206\OneDrive - BNSF Railway\RoboRailCop\2025-01-16_all_trnv_images\crops\cont_face"

# Create a list of class identifiers
class_identifiers = ['1', '2', '3', '4', '5']  # Add more class identifiers as needed


def resize_image(image, max_width=1920, max_height=1080):
    """Resizes the image to fit within the specified dimensions while maintaining aspect ratio."""
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


# Process each image in the source directory
for image_name in os.listdir(source_dir):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(source_dir, image_name)

        # Load the image
        image = cv2.imread(image_path)

        # Resize the image for display
        resized_image = resize_image(image)

        # Display the image
        cv2.imshow('Image', resized_image)
        cv2.waitKey(1)  # Display the image for a short moment to ensure it appears

        # Prompt the user for a class identifier
        class_id = input(f"Enter class identifier for {image_name} ({', '.join(class_identifiers)}): ")

        # Validate the input
        if class_id in class_identifiers:
            if class_id == '1':
                target_dir = closed_door_dir
            elif class_id == '2':
                target_dir = container_side_dir
            elif class_id == '3':
                target_dir = open_door_dir
            elif class_id == '4':
                target_dir = handhold_dir
            elif class_id == '5':
                target_dir = cont_face_dir

            # Move the image to the corresponding class directory
            target_path = os.path.join(target_dir, image_name)
            shutil.move(image_path, target_path)
            print(f"Moved {image_name} to class {class_id}")
        else:
            print(f"Invalid class identifier for {image_name}. Skipping...")

        # Close the image window
        cv2.destroyAllWindows()

print('Dataset organization complete.')