# 🩺 Multi Cancer Detection using Convolutional Neural Networks (CNN)

A deep learning project developed using **PyTorch** to classify multiple cancer types from medical images using a custom Convolutional Neural Network (CNN).

The model is trained on a multi-cancer image dataset and is capable of predicting **26 different cancer classes** from a single input image.

---

# Features

- Custom CNN built completely from scratch
- Supports **26 cancer classes**
- Automatic dataset flattening
- PyTorch DataLoader pipeline
- GPU acceleration (CUDA)
- Model checkpoint saving
- Training history stored as JSON
- Training loss visualization
- Prediction on custom images
- Modular project structure

---

# Project Structure

```
MULTI_CANCER_DETECTION
│
├── assets/
├── notebooks/
│
├── saved_models/
│   └── cancer_cnn.pth
│
├── test_images/
│   └── image (1).webp
│
├── dataset.py
├── model.py
├── train.py
├── predict.py
├── config.py
├── utils.py
│
├── training_history.json
├── training_history.png
│
├── requirements.txt
└── README.md
```

---

# Dataset

The project uses a Multi Cancer Dataset consisting of the following categories:

## Acute Lymphoblastic Leukemia (ALL)

- all_benign
- all_early
- all_pre
- all_pro

## Brain Cancer

- brain_glioma
- brain_menin
- brain_tumor

## Breast Cancer

- breast_benign
- breast_malignant

## Cervical Cancer

- cervix_dyk
- cervix_koc
- cervix_mep
- cervix_pab
- cervix_sfi

## Kidney Cancer

- kidney_normal
- kidney_tumor

## Lung and Colon Cancer

- colon_aca
- colon_bnt
- lung_aca
- lung_bnt
- lung_scc

## Lymphoma

- lymph_cll
- lymph_fll
- lymph_mcl

## Oral Cancer

- oral_normal
- oral_scc

Total Classes : **26**

---

# Model Architecture

The CNN consists of:

Input Image (224 × 224 × 3)

↓

Conv2D (3 → 16)

↓

ReLU

↓

MaxPool

↓

Conv2D (16 → 32)

↓

ReLU

↓

MaxPool

↓

Flatten

↓

Fully Connected (128)

↓

Output Layer (26 Classes)

---

# Installation

Clone the repository

```bash
git clone https://github.com/Manav0401/Multi_Cancer_Detection.git

cd Multi_Cancer_Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Run

```bash
python train.py
```

During training the project automatically

- Loads the dataset
- Trains the CNN
- Saves the trained model
- Saves training history
- Generates training graphs

Saved files

```
saved_models/
    cancer_cnn.pth

training_history.json
training_history.png
```

---

# Training Loss

<p align="center">
  <img src="/training_history.png" width="700">
</p>

---

# Prediction

Place your image inside

```
test_images/
```

Update the filename inside

```python
predict.py
```

Run

```bash
python predict.py
```

Example Output

```
Prediction Result

Image       : image.webp
Prediction  : breast_malignant
Confidence  : 98.72%
```

---

# Training History

The training history is automatically stored as

```
training_history.json
```

Example

```json
{
    "train_loss": [
        0.5876,
        0.2664,
        0.1722
    ],
    "epoch_time": [
        395.02,
        390.98,
        390.38
    ]
}
```

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow
- Matplotlib
- tqdm

---

# Future Improvements

- Validation Dataset
- Accuracy & Loss Curves
- Early Stopping
- Learning Rate Scheduler
- Transfer Learning
- Grad-CAM Visualization
- Web Application (Flask / Streamlit)
- REST API Deployment

---

# Author

**Manav M George**

Integrated M.Tech Artificial Intelligence

VIT Bhopal University

---

## Medical Disclaimer

This project is intended for educational and research purposes only.

It is **not a certified medical diagnostic system** and should not be used for clinical diagnosis, treatment, or medical decision-making.

Always consult qualified healthcare professionals for medical advice and diagnosis.