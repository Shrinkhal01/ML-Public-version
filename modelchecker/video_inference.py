import cv2
import tensorflow as tf
import numpy as np
import os

# Load your trained model from the saved location.
# Adjust the path relative to this script's location.
model = tf.keras.models.load_model("../saved_model/my_model.keras")

# Define image parameters (same as used during training)
img_height = 720
img_width = 1280

# Generate class names based on folder names in the train directory.
# This excludes hidden files like .DS_Store.
train_dir = os.path.join("..", "train")
class_names = [d for d in os.listdir(train_dir) if not d.startswith('.')]
class_names.sort()

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (img_width, img_height))
    frame_array = np.array(frame_resized, dtype=np.float32)
    frame_array = np.expand_dims(frame_array, axis=0)
    return frame_array
video_source = ""  # or 0 to use a webcam feed
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break

    # Preprocess the frame for the model.
    preprocessed_frame = preprocess_frame(frame)

    # Run model inference on the frame.
    predictions = model.predict(preprocessed_frame)
    pred_class = np.argmax(predictions[0])
    confidence = predictions[0][pred_class]

    # Build the text to display on the frame.
    label_text = f"{class_names[pred_class]}: {confidence:.2f}"

    # Display the prediction on the frame.
    # You can adjust position, font, color, and thickness as needed.
    cv2.putText(frame, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Accident Detection", frame)

    # Exit if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
