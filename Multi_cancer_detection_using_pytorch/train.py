import time

import torch
import torch.nn as nn
from tqdm import tqdm

from config import DEVICE, LEARNING_RATE, NUM_EPOCHS, MODEL_PATH, HISTORY_PATH
from dataset import train_loader
from model import CancerCNN
from utils import save_model, save_history


model = CancerCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

history = {
    "train_loss": [],
    "epoch_time": []
}


for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0

    start = time.time()

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"
    )

    for images, labels in progress:

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress.set_postfix(
            Loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / len(train_loader)
    epoch_time = time.time() - start

    history["train_loss"].append(epoch_loss)
    history["epoch_time"].append(epoch_time)

    print(
        f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Average Loss: {epoch_loss:.4f} "
        f"Time: {epoch_time:.2f}s"
    )


save_model(model, MODEL_PATH)
save_history(history, HISTORY_PATH)

print("\nTraining Completed Successfully!")
print(f"Model Saved   : {MODEL_PATH}")
print(f"History Saved : {HISTORY_PATH}")