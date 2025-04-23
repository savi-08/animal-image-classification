import tensorflow as tf
import numpy as np
import cv2
import os

# Load trained model
model = tf.keras.models.load_model("animal_classifier_model.h5")

# List class names (should match folder names in dataset/)
class_names = sorted(os.listdir("dataset"))

# Ask user for image file name
img_path = input("Enter the image filename (e.g., test_cat.jpg): ")

# Check if file exists
if not os.path.exists(img_path):
    print(f"❌ File '{img_path}' not found.")
    exit()

# Load and preprocess image
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(img)
predicted_class = class_names[np.argmax(pred)]
print(f"✅ Prediction: {predicted_class}")
