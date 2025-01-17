import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# Define the CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(5, activation='softmax')  # 5 output classes
    ])
    return model


# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values between 0 and 1
    rotation_range=20,      # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift horizontally
    height_shift_range=0.2, # Randomly shift vertically
    zoom_range=0.2,         # Randomly zoom in/out
    horizontal_flip=True,   # Randomly flip images horizontally
)

# Validation data (no augmentation, just rescaling)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation datasets
train_data = train_datagen.flow_from_directory(
    r"C:\Users\c883206\OneDrive - BNSF Railway\RoboRailCop\2025-01-16_all_trnv_images\dataset\train",
    target_size=(128, 128),  # Resize images to 128x128
    batch_size=32,
    class_mode='categorical',  # For multi-class classification
)

validation_data = validation_datagen.flow_from_directory(
    r"C:\Users\c883206\OneDrive - BNSF Railway\RoboRailCop\2025-01-16_all_trnv_images\dataset\validation",
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
)

model = create_model()
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=20,  # Number of training epochs
)

# Test dataset
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    r"C:\Users\c883206\OneDrive - BNSF Railway\RoboRailCop\2025-01-16_all_trnv_images\dataset\test",
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.show()

model.save('5_class_cnn_model.h5')