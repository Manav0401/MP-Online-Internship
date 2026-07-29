from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

model = joblib.load("model/best_model.pkl")
df = pd.read_csv("data/cleaned_dataset.csv")


def get_dropdown_data():
    return {
        "brands": sorted(df["brand"].unique()),
        "fuel_types": sorted(df["fuel_type"].unique()),
        "transmission_types": sorted(df["transmission_type"].unique()),
        "seller_types": sorted(df["seller_type"].unique())
    }


@app.route("/")
def home():

    return render_template(
    "index.html",
    **get_dropdown_data(),
    prediction=None,
    selected_brand="",
    selected_model="",
    selected_fuel="",
    selected_transmission="",
    selected_seller="",
    manufacturing_year="",
    km_driven="",
    mileage="",
    engine="",
    max_power="",
    seats=""
)

@app.route("/get_models/<brand>")
def get_models(brand):

    models = sorted(
        df[df["brand"] == brand]["model"].unique().tolist()
    )

    return jsonify(models)


@app.route("/get_specs/<brand>/<model_name>")
def get_specs(brand, model_name):

    car = df[
        (df["brand"] == brand) &
        (df["model"] == model_name)
    ]

    if car.empty:
        return jsonify({})

    car = car.iloc[0]

    return jsonify({
        "engine": float(car["engine"]),
        "mileage": float(car["mileage"]),
        "max_power": float(car["max_power"]),
        "seats": int(car["seats"])
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        brand = request.form["brand"]
        model_name = request.form["model"]

        manufacturing_year = int(request.form["manufacturing_year"])
        vehicle_age = datetime.now().year - manufacturing_year

        km_driven = float(request.form["km_driven"])

        seller_type = request.form["seller_type"]
        fuel_type = request.form["fuel_type"]
        transmission_type = request.form["transmission_type"]

        mileage = float(request.form["mileage"])
        engine = float(request.form["engine"])
        max_power = float(request.form["max_power"])
        seats = int(request.form["seats"])

        input_data = pd.DataFrame([{
            "brand": brand,
            "model": model_name,
            "vehicle_age": vehicle_age,
            "km_driven": km_driven,
            "seller_type": seller_type,
            "fuel_type": fuel_type,
            "transmission_type": transmission_type,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats
        }])

        predicted_price = model.predict(input_data)[0]
        prediction = f"₹ {predicted_price:,.0f}"

    except Exception as e:

        prediction = f"Prediction Error: {str(e)}"

    return render_template(
    "index.html",
    **get_dropdown_data(),
    prediction=prediction,

    selected_brand=brand,
    selected_model=model_name,
    selected_fuel=fuel_type,
    selected_transmission=transmission_type,
    selected_seller=seller_type,

    manufacturing_year=manufacturing_year,
    km_driven=km_driven,

    mileage=mileage,
    engine=engine,
    max_power=max_power,
    seats=seats
)


if __name__ == "__main__":
    app.run(debug=True)