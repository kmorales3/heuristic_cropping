from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = load_model(r'C:\Users\c883206\OneDrive - BNSF Railway\RoboRailCop\heuristic_cropping_attempt\5_class_cnn_model.h5')


def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocesses the input image:
    - Loads the image from the given path.
    - Resizes it to the target size.
    - Normalizes pixel values between 0 and 1.

    Returns:
        The preprocessed image as a NumPy array ready for model inference.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image to the target size
    image = cv2.resize(image, target_size)

    # Convert the image to a NumPy array
    image = np.array(image)

    # Normalize the image (scale pixel values to [0, 1])
    image = image / 255.0

    # Add an extra dimension to match the model's expected input shape (batch size, height, width, channels)
    image = np.expand_dims(image, axis=0)  # Shape becomes (1, 128, 128, 3)

    return image


# Path to the image you want to classify
image_path = r"C:\Users\c883206\OneDrive - BNSF Railway\RoboRailCop\2025-01-16_all_trnv_images\dataset\test\open_door\cropped_region_39743_91.jpg"

# Preprocess the image
input_image = preprocess_image(image_path)

# Perform inference
predictions = model.predict(input_image)

# The output of `model.predict` will be a probability distribution for each class
print("Predictions:", predictions)

# Get the class with the highest probability
predicted_class = np.argmax(predictions, axis=1)  # Index of the highest probability
print(f"Predicted Class: {predicted_class}")

# Define your class labels (these must match the order used during training)
class_labels = ['closed door', 'container face', 'container side', 'handholds', 'open door']

# Get the predicted class label
predicted_label = class_labels[predicted_class[0]]
print(f"Predicted Label: {predicted_label}")
