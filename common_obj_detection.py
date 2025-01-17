#!"c:/Users/c883206/OneDrive - BNSF Railway/RoboRailCop/heuristic_cropping_attempt/.venv/Scripts/python"
import cv2
from ultralytics import YOLO
import os

# Load the YOLO model
model = YOLO("yolov8n.pt")  # You can specify different YOLO models here

crop_location = r"C:\Users\c883206\OneDrive - BNSF Railway\RoboRailCop\2025-01-16_all_trnv_images\crops"

for crop in os.listdir(crop_location):
    if crop.endswith(('.jpg', '.jpeg', '.png')):
        # Load an image
        img = cv2.imread(os.path.join(crop_location, crop))  # Replace with your image path

        # Perform object detection
        results = model.predict(source=img)  # Use 'cap' for video

        # Process the results
        detection_made = False
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection_made = True
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Extract class label and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Draw bounding box and label on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{model.names[class_id]} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Only display the image if a detection was made
        if detection_made:
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()