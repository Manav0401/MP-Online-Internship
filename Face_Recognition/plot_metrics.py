import json
import matplotlib.pyplot as plt

with open("history.json","r") as f:

    history=json.load(f)

plt.figure(figsize=(8,5))

plt.plot(history["train_accuracy"],label="Train")

plt.plot(history["validation_accuracy"],label="Validation")

plt.legend()

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.savefig("results/accuracy.png")

plt.show()

plt.figure(figsize=(8,5))

plt.plot(history["train_loss"],label="Train")

plt.plot(history["validation_loss"],label="Validation")

plt.legend()

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.savefig("results/loss.png")

plt.show()