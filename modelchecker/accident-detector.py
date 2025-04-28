import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import numpy as np
import os
import cv2

from email_sender import notify_authorities

MODEL_PATH = "../saved_model/my_model.keras"
TRAIN_DIR = "../train"
VIDEO_PATH = "/Users/shrinkhals/Downloads/accidentl.mp4"
IMG_HEIGHT = 720
IMG_WIDTH = 1280

SCREENSHOTS_FOLDER = "screenshots"
os.makedirs(SCREENSHOTS_FOLDER, exist_ok=True)

def get_class_names(train_directory):
    return sorted([d for d in os.listdir(train_directory) if not d.startswith('.')])
def preprocess_frame(frame, target_size):
    frame_resized = cv2.resize(frame, target_size)
    frame_array = img_to_array(frame_resized)
    return np.expand_dims(frame_array, axis=0) 
def process_video(video_path, model, class_names, target_size):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    accident_counter = 0
    accident_frames = []
    email_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_array = preprocess_frame(frame, target_size)
        predictions = model.predict(frame_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        # Debugging: Print predicted class and confidence
        print(f"Predicted class: {class_names[predicted_class]}, Confidence: {confidence:.2f}")

        label = f"{class_names[predicted_class]} ({confidence:.2f})"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Check if the detected class is "Accident" (case-insensitive) and confidence > 0.85
        if class_names[predicted_class].lower() == "accident" and confidence >= 0.85:
            accident_counter += 1

            # Turn the screen boundary red
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)

            # Save the screenshot in the screenshots folder
            screenshot_path = os.path.join(SCREENSHOTS_FOLDER, f"accident_frame_{accident_counter}.jpg")
            if cv2.imwrite(screenshot_path, frame):
                print(f"Screenshot saved: {screenshot_path}")
            else:
                print(f"Failed to save screenshot: {screenshot_path}")

            accident_frames.append(screenshot_path)

            if accident_counter == 4:
                if email_counter < 2:
                    print("4 consecutive accident frames detected. Sending email...")
                    notify_authorities(
                        subject="Accident Detected!",
                        body="An accident has been detected. Please find the attached screenshots.",
                        attachments=accident_frames
                    )
                    print("Email sent with accident screenshots.")
                    email_counter += 1
                else:
                    print("Email limit reached. No more emails will be sent for this video.")

                accident_counter = 0
                accident_frames = []

                if email_counter == 2:
                    print("Email limit reached. Exiting program.")
                    break
        else:
            # Reset counters if no accident is detected
            print("No accident detected. Resetting counters.")
            accident_counter = 0
            accident_frames = []

        # Display the frame with the red boundary (if accident detected)
        cv2.imshow("Video", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting detection as 'q' was pressed.")
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = get_class_names(TRAIN_DIR)
    process_video(VIDEO_PATH, model, class_names, (IMG_WIDTH, IMG_HEIGHT))