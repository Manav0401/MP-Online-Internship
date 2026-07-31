import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set Random Seeds (for reproducibility)

tf.random.set_seed(123)
np.random.seed(123)

# Dataset Path

dataset_path = "dataset/raw-img"      # Change this if needed

if not os.path.exists(dataset_path):
    raise FileNotFoundError(
        f"Dataset not found at: {dataset_path}"
    )

# Load Dataset

train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256, 256),
    batch_size=32
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=32
)

class_names = train_dataset.class_names

print("\nClasses:")
print(class_names)

# Optimize Dataset

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# Data Augmentation

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Load ResNet50

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(256, 256, 3)
)

base_model.trainable = False

# ---------------------------------------------------
# Build Model
# ---------------------------------------------------

inputs = tf.keras.Input(shape=(256, 256, 3))

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

x = layers.Dense(256, activation="relu")(x)

outputs = layers.Dense(10, activation="softmax")(x)

model = Model(inputs, outputs)

print("\nModel Summary:\n")
model.summary()

# Compile Model

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Create Model Save Folder

os.makedirs("saved_model", exist_ok=True)

# Callbacks

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath="saved_model/resnet50_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# Train Model

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)

# Save Training History

with open("history.json", "w") as f:
    json.dump(history.history, f)

print("\nTraining history saved as history.json")

print("\nBest model saved to:")
print("saved_model/resnet50_best.keras")

print("\nTraining completed successfully!")
