import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

# Load model
model = tf.keras.models.load_model("../saved_model/my_model.keras")

# Image properties (same as training)
img_height = 720
img_width = 1280

# Load and preprocess the image
img_path = "check1.jpeg"
img = load_img(img_path, target_size=(img_height, img_width))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

# Get class names from training directory
class_names = [d for d in os.listdir("../train") if not d.startswith('.')]
class_names.sort()

# Print result
print(f"Predicted class: {class_names[predicted_class]} (confidence: {predictions[0][predicted_class]:.2f})")
