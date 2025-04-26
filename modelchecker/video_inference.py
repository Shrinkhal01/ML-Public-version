import cv2
import tensorflow as tf
import numpy as np
import os

# Constants
MODEL_PATH = "../saved_model/my_model.keras"
TRAIN_DIR = "../train"
VIDEO_SOURCE = "acc8.mp4"  # Change to 0 for webcam feed
IMG_HEIGHT = 720
IMG_WIDTH = 1280

def get_class_names(train_directory):
    """Get sorted class names from the training directory."""
    return sorted([d for d in os.listdir(train_directory) if not d.startswith('.')])

def preprocess_frame(frame, target_size):
    """Resize and preprocess a frame for model prediction."""
    frame_resized = cv2.resize(frame, target_size)
    frame_array = np.array(frame_resized, dtype=np.float32)
    return np.expand_dims(frame_array, axis=0)

def process_video(video_source, model, class_names, target_size):
    """Process video frame by frame and display predictions."""
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read the frame.")
            break

        # Preprocess frame and predict
        preprocessed_frame = preprocess_frame(frame, target_size)
        predictions = model.predict(preprocessed_frame)
        pred_class = np.argmax(predictions[0])
        confidence = predictions[0][pred_class]

        # Display prediction on the frame
        label_text = f"{class_names[pred_class]}: {confidence:.2f}"
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Accident Detection", frame)

        # Exit on 'q' press
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
    process_video(VIDEO_SOURCE, model, class_names, (IMG_WIDTH, IMG_HEIGHT))