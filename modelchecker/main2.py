import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import numpy as np
import os
import cv2  # OpenCV for video processing

# Load model
model = tf.keras.models.load_model("../saved_model/my_model.keras")

# Image properties (same as training)
img_height = 720
img_width = 1280

# Get class names from training directory
class_names = [d for d in os.listdir("../train") if not d.startswith('.')]
class_names.sort()

# Video file path
video_path = "/Users/shrinkhals/Downloads/videoa 6.MOV"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match model input size
    frame_resized = cv2.resize(frame, (img_width, img_height))
    frame_array = img_to_array(frame_resized)
    frame_array = np.expand_dims(frame_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(frame_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Display prediction on the frame
    label = f"{class_names[predicted_class]} ({confidence:.2f})"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Video", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()