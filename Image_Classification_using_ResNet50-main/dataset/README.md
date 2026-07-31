# Dataset

This project uses the **Animals-10** dataset from Kaggle.

Download the dataset from:

https://www.kaggle.com/datasets/alessiocorrado99/animals10

Or using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download("alessiocorrado99/animals10")

print("Dataset downloaded to:", path)
```

After downloading, place the `raw-img` folder inside this `dataset` directory.

Expected structure:

dataset/
└── raw-img/
    ├── cane/
    ├── cavallo/
    ├── elefante/
    ├── ...