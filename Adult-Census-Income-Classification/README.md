# Adult Census Income Classification

## Objective

The objective of this project is to build and compare multiple machine learning classification models to predict whether an individual's annual income exceeds \$50,000 based on demographic and employment-related attributes. The project also aims to identify the best-performing model using standard evaluation metrics.

---

## Dataset

- **Dataset Name:** Adult Census Income Dataset (Kaggle)
- **Target Variable:** `income`
  - `<=50K`
  - `>50K`

The dataset contains demographic and employment-related information such as age, education, occupation, work class, marital status, hours worked per week, and capital gain/loss.

---

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Methodology

### 1. Dataset Understanding
- Loaded the Adult Census Income dataset.
- Displayed the first five records.
- Examined dataset shape and information.
- Identified numerical and categorical features.
- Analyzed the target variable distribution.
- Checked for missing values.

### 2. Data Cleaning
- Identified missing values represented by `?`.
- Replaced `?` with `NaN`.
- Removed rows containing missing values.
- Removed duplicate records.
- Verified the cleaned dataset.

### 3. Feature Engineering
- Encoded all categorical features using Label Encoding.
- Separated features and target variable.
- Split the dataset into 80% training and 20% testing sets.
- Standardized features using StandardScaler for models requiring feature scaling.

### 4. Model Building
The following machine learning models were implemented:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

### 5. Model Evaluation
Each model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

The best-performing model was selected based on these evaluation metrics.

---

## Results

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-----------|---------:|----------:|-------:|---------:|--------:|
| Logistic Regression | 0.8238 | 0.6747 | 0.4767 | 0.5568 | 0.8460 |
| Decision Tree | 0.8119 | 0.6008 | 0.6324 | 0.6161 | 0.7492 |
| **Random Forest** | **0.8550** | **0.7267** | **0.6354** | **0.6780** | **0.9049** |
| KNN | 0.8253 | 0.6438 | 0.5992 | 0.6205 | 0.8432 |
| SVM | 0.8462 | 0.7416 | 0.5539 | 0.6326 | 0.8874 |

### Best Performing Model

**Random Forest Classifier**

- Accuracy: **85.50%**
- Precision: **72.67%**
- Recall: **63.54%**
- F1 Score: **67.80%**
- ROC-AUC: **90.49%**

---

## Observations

1. Random Forest achieved the highest overall performance with an accuracy of **85.50%** and a ROC-AUC score of **90.49%**.
2. SVM produced the second-best accuracy, demonstrating strong classification performance after feature scaling.
3. Logistic Regression provided competitive results despite being a simpler linear model.
4. Decision Tree showed comparatively lower performance and was more prone to overfitting than the ensemble-based Random Forest model.
5. Feature scaling significantly improved the performance of KNN and SVM classifiers.

---

## Conclusion

This project successfully developed and compared five machine learning classification models for predicting annual income using the Adult Census Income dataset. After preprocessing, feature engineering, and model evaluation, the Random Forest Classifier achieved the best overall performance with an accuracy of **85.50%** and a ROC-AUC score of **90.49%**. The results demonstrate that ensemble learning methods are highly effective for this classification task and provide better predictive performance than individual classifiers. This project highlights the complete machine learning workflow, from data preprocessing to model evaluation and selection.

---

## Project Structure

```
Adult-Census-Income-Classification/
│
├── dataset/
│   └── adult.csv
│
├── train.py
├── README.md
└── results/
```

---

## Author

**Manav M George**

Integrated M.Tech (Artificial Intelligence)

VIT Bhopal University