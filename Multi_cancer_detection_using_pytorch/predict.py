import os

import torch
from PIL import Image
from torchvision import transforms

from config import DEVICE, MODEL_PATH, IMAGE_SIZE, TEST_IMAGE_DIR
from model import CancerCNN
from utils import load_model


class_names = [
    "all_benign",
    "all_early",
    "all_pre",
    "all_pro",

    "brain_glioma",
    "brain_menin",
    "brain_tumor",

    "breast_benign",
    "breast_malignant",

    "cervix_dyk",
    "cervix_koc",
    "cervix_mep",
    "cervix_pab",
    "cervix_sfi",

    "colon_aca",
    "colon_bnt",

    "kidney_normal",
    "kidney_tumor",

    "lung_aca",
    "lung_bnt",
    "lung_scc",

    "lymph_cll",
    "lymph_fll",
    "lymph_mcl",

    "oral_normal",
    "oral_scc"
]


transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])


model = CancerCNN().to(DEVICE)
model = load_model(model, MODEL_PATH, DEVICE)


image_path = os.path.join(
    TEST_IMAGE_DIR,
    "image.webp"
)


image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0).to(DEVICE)


with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, dim=1)


print("\nPrediction Result")
print("----------------------------")
print(f"Image      : {os.path.basename(image_path)}")
print(f"Prediction : {class_names[predicted.item()]}")
print(f"Confidence : {confidence.item()*100:.2f}%")