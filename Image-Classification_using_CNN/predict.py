import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Load Trained Model

model = load_model("saved_model/cnn_best.keras")

print("Model Loaded Successfully!")

# Class Names

class_names = [
    "cane",
    "cavallo",
    "elefante",
    "farfalla",
    "gallina",
    "gatto",
    "mucca",
    "pecora",
    "ragno",
    "scoiattolo"
]

# Test Image

image_path = "test/download.jpg"      # Change image name if needed

# Load Image

img = load_img(image_path, target_size=(180, 180))

plt.figure(figsize=(5,5))
plt.imshow(img)
plt.axis("off")
plt.show()

# Preprocess Image

img_array = img_to_array(img)

img_array = img_array.astype("float32") / 255.0

img_array = np.expand_dims(img_array, axis=0)

# Predict

prediction = model.predict(img_array, verbose=0)

predicted_index = np.argmax(prediction)

confidence = prediction[0][predicted_index] * 100

# Print Prediction Probabilities

print("\nPrediction Probabilities:\n")

for i, probability in enumerate(prediction[0]):
    print(f"{class_names[i]:12} : {probability*100:.2f}%")

print("\n-----------------------------")

print(f"Predicted Class : {class_names[predicted_index]}")

print(f"Confidence      : {confidence:.2f}%")