import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import numpy as np
import os
import cv2  # OpenCV for video processing

# Constants
MODEL_PATH = "../saved_model/my_model.keras"
TRAIN_DIR = "../train"
VIDEO_PATH = "acc8.mp4"
IMG_HEIGHT = 720
IMG_WIDTH = 1280

def get_class_names(train_directory):
    """Get sorted class names from the training directory."""
    return sorted([d for d in os.listdir(train_directory) if not d.startswith('.')])

def preprocess_frame(frame, target_size):
    """Resize and preprocess a frame for model prediction."""
    frame_resized = cv2.resize(frame, target_size)
    frame_array = img_to_array(frame_resized)
    return np.expand_dims(frame_array, axis=0)  # Add batch dimension

def process_video(video_path, model, class_names, target_size):
    """Process video frame by frame and display predictions."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame and predict
        frame_array = preprocess_frame(frame, target_size)
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

if __name__ == "__main__":
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Get class names
    class_names = get_class_names(TRAIN_DIR)

    # Process video
    process_video(VIDEO_PATH, model, class_names, (IMG_WIDTH, IMG_HEIGHT))