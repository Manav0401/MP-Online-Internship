import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load Model

model = tf.keras.models.load_model("saved_model/resnet50_best.keras")

# Load Class Names


class_names = sorted(os.listdir("dataset/raw-img"))

# Load Image

image_path = "test/download.jpg" # or you can use "image_path = input("Enter image path: ")"

img = load_img(image_path, target_size=(256, 256))

plt.imshow(img)
plt.axis("off")
plt.show()

# Preprocess Image

img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Predict

prediction = model.predict(img_array, verbose=0)

predicted_index = np.argmax(prediction)
confidence = np.max(prediction) * 100

print(f"Prediction : {class_names[predicted_index]}")
print(f"Confidence : {confidence:.2f}%")

# Show Probabilities

print("\nPrediction Probabilities:")

for name, prob in zip(class_names, prediction[0]):
    print(f"{name:<12}: {prob*100:.2f}%")
