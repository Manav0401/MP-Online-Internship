import json
import matplotlib.pyplot as plt


with open("training_history.json", "r") as f:
    history = json.load(f)


train_loss = history["train_loss"]
epoch_time = history["epoch_time"]

epochs = range(1, len(train_loss) + 1)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, marker="o", linewidth=2)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, epoch_time, marker="o", linewidth=2)
plt.title("Epoch Time")
plt.xlabel("Epoch")
plt.ylabel("Time (seconds)")
plt.grid(True)

plt.tight_layout()
plt.savefig("training_history.png", dpi=300)
plt.show()