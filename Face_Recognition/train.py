import os
import json
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

print("PyTorch Version:", torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

dataset_path = "/kaggle/input/datasets/jessicali9530/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled"

filtered_dataset = "/kaggle/working/filtered_faces"

MIN_IMAGES = 40

if os.path.exists(filtered_dataset):
    shutil.rmtree(filtered_dataset)

os.makedirs(filtered_dataset)

selected_people = []

for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    images = os.listdir(person_path)

    if len(images) >= MIN_IMAGES:

        shutil.copytree(
            person_path,
            os.path.join(filtered_dataset, person)
        )

        selected_people.append(person)

print("People Selected:", len(selected_people))

train_transform = transforms.Compose([

    transforms.Resize((160,160)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.08,0.08)
    ),

    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25
    ),

    transforms.ToTensor()

])

validation_transform = transforms.Compose([

    transforms.Resize((160,160)),

    transforms.ToTensor()

])

full_dataset = datasets.ImageFolder(
    filtered_dataset,
    transform=train_transform
)

class_names = full_dataset.classes

print(class_names)

train_size = int(0.8 * len(full_dataset))
validation_size = len(full_dataset) - train_size

train_dataset, validation_dataset = random_split(
    full_dataset,
    [train_size, validation_size]
)

validation_dataset.dataset.transform = validation_transform

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False
)

print("Training Images:", len(train_dataset))
print("Validation Images:", len(validation_dataset))

class FaceCNN(nn.Module):

    def __init__(self, num_classes):

        super(FaceCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256,num_classes)

        )

    def forward(self,x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = FaceCNN(len(class_names)).to(device)

print(model)


criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

epochs = 20

best_accuracy = 0

history = {

    "train_loss": [],
    "train_accuracy": [],
    "validation_loss": [],
    "validation_accuracy": []

}

with open("/kaggle/working/history.json", "w") as f:

    json.dump(history, f)

print("History Saved!")