import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("data/cleaned_dataset.csv")

X = df.drop("selling_price", axis=1)
y = df["selling_price"]

categorical_features = [
    "brand",
    "model",
    "seller_type",
    "fuel_type",
    "transmission_type"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []

best_model = None
best_score = -1
best_name = ""

for name, regressor in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    results.append([name, r2, mae, rmse])

    if r2 > best_score:
        best_score = r2
        best_model = pipeline
        best_name = name

results_df = pd.DataFrame(
    results,
    columns=["Model", "R² Score", "MAE", "RMSE"]
)

print("Model Comparison")

print(results_df.to_string(index=False))

joblib.dump(best_model, "model/best_model.pkl")

print(f"Best Model : {best_name}")
print(f"R² Score   : {best_score:.4f}")
print("Model saved as model/best_model.pkl")