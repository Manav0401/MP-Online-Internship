import os
import shutil

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Dataset Paths

SOURCE_PATH = "OriginalDataset"
DESTINATION_PATH = "FlattenedDataset"

# Flatten Dataset (Only Once)

if not os.path.exists(DESTINATION_PATH):

    os.makedirs(DESTINATION_PATH)

    total_copied = 0

    for root, _, files in os.walk(SOURCE_PATH):

        image_files = [
            f for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            continue

        class_name = os.path.basename(root)

        class_folder = os.path.join(
            DESTINATION_PATH,
            class_name
        )

        os.makedirs(class_folder, exist_ok=True)

        for image in image_files:

            src = os.path.join(root, image)
            dst = os.path.join(class_folder, image)

            shutil.copy2(src, dst)

            total_copied += 1

    print("\n✅ Dataset Flattened Successfully!")
    print(f"📸 Total Images Copied: {total_copied}")

else:

    print("✅ Flattened dataset already exists. Skipping...")

# Image Transformations

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load Dataset

dataset = ImageFolder(
    root=DESTINATION_PATH,
    transform=transform
)

# DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# Dataset Information

class_names = dataset.classes
num_classes = len(class_names)

print(f"\n Dataset Loaded Successfully!")
print(f" Number of Classes : {num_classes}")
print(f" Total Images      : {len(dataset)}")
print(" Batch Size        : 64")