import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


df = pd.read_csv("dataset/adult.csv")

print("First Five Records:\n")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nDataset Information:\n")
df.info()
print("\nMissing Values:\n")
print(df.isnull().sum())
print("\nStatistical Summary:\n")
print(df.describe())
print("\nNumerical Features:")
print(df.select_dtypes(include=["int64", "float64"]).columns.tolist())

print("\nCategorical Features:")
print(df.select_dtypes(include=["object", "string"]).columns.tolist())
print("\nTarget Variable Distribution:\n")
print(df["income"].value_counts())

print("\nMissing Values represented by '?'\n")

columns = ["workclass", "occupation", "native.country"]

for col in columns:
    print(f"{col}: {(df[col] == '?').sum()}")
df.replace('?', np.nan, inplace=True)

print("\nMissing Values After Replacement:\n")
print(df.isnull().sum())

df.dropna(inplace=True)

print("\nDataset Shape After Removing Missing Values:")

print(df.shape)

duplicates = df.duplicated().sum()

print("\nDuplicate Rows:", duplicates)

df.drop_duplicates(inplace=True)

print("Shape After Removing Duplicates:")

print(df.shape)

print("\nRemaining Missing Values:\n")

print(df.isnull().sum())

label_encoder = LabelEncoder()
categorical_columns = df.select_dtypes(include=["object", "string"]).columns
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

print("\nEncoded Dataset (First Five Records):\n")
print(df.head())

X = df.drop("income", axis=1)
y = df["income"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed!")

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)

svm_model = SVC(kernel='rbf', random_state=42)

svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)

def evaluate_model(name, model, X_test, y_test, y_pred):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    return {
        "Algorithm": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }
    
results = []

results.append(
    evaluate_model(
        "Logistic Regression",
        log_model,
        X_test_scaled,
        y_test,
        y_pred_log,
    )
)

results.append(
    evaluate_model(
        "Decision Tree",
        dt_model,
        X_test,
        y_test,
        y_pred_dt,
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        rf_model,
        X_test,
        y_test,
        y_pred_rf,
    )
)

results.append(
    evaluate_model(
        "KNN",
        knn_model,
        X_test_scaled,
        y_test,
        y_pred_knn,
    )
)

results.append(
    evaluate_model(
        "SVM",
        svm_model,
        X_test_scaled,
        y_test,
        y_pred_svm,
    )
)

results_df = pd.DataFrame(results)

print("\nModel Comparison:\n")
print(results_df)

best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\nBest Performing Model:\n")
print(best_model)