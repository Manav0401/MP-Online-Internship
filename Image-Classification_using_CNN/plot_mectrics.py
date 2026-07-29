import json
import matplotlib.pyplot as plt
import os

# =======================================
# Create Results Folder
# =======================================

os.makedirs("results", exist_ok=True)

# =======================================
# Load History
# =======================================

with open("history.json", "r") as f:
    history = json.load(f)

# =======================================
# Accuracy Plot
# =======================================

plt.figure(figsize=(8,5))

plt.plot(history["accuracy"], label="Training")

plt.plot(history["val_accuracy"], label="Validation")

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("results/accuracy.png")

plt.show()

# =======================================
# Loss Plot
# =======================================

plt.figure(figsize=(8,5))

plt.plot(history["loss"], label="Training")

plt.plot(history["val_loss"], label="Validation")

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("results/loss.png")

plt.show()

print("Graphs saved successfully!")