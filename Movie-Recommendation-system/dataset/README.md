# Dataset

This project uses the **MovieLens 25M Dataset** for training the movie recommendation model.

> **Note:** The dataset is **not included** in this repository

## Download the Dataset

Download the dataset from Kaggle:

**MovieLens 25M Dataset**  
https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system

## Required Files

After downloading, place the following files inside this folder:

```text
dataset/
├── movies.csv
└── ratings.csv
```

## Train the Model

Once the dataset has been added, generate the recommendation model by running:

```bash
python train.py
```

This will create the required model files in the `model/` directory:

```text
model/
├── similarity.pkl
├── movies.pkl
└── movie_list.pkl
```