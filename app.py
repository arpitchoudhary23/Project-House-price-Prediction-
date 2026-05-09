from flask import Flask, request, render_template
import pickle
import pandas as pd
import sqlite3
from database import init_db

init_db()

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        input_data = pd.DataFrame([{
            "Area": float(data["area"]),
            "Bedrooms": float(data["bedrooms"]),
            "Bathrooms": float(data["bathrooms"]),
            "Floors": float(data["floors"]),
            "YearBuilt": float(data["yearbuilt"]),
            "Location": data["location"],
            "Condition": data["condition"],
            "Garage": data["garage"]
        }])

        price = model.predict(input_data)[0]

        conn = sqlite3.connect("predictions.db")
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO predictions (
            area, bedrooms, bathrooms, floors, yearbuilt,
            location, condition, garage, predicted_price
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            input_data["Area"][0],
            input_data["Bedrooms"][0],
            input_data["Bathrooms"][0],
            input_data["Floors"][0],
            input_data["YearBuilt"][0],
            input_data["Location"][0],
            input_data["Condition"][0],
            input_data["Garage"][0],
            price
        ))

        conn.commit()
        conn.close()

        return render_template(
            "index.html",
            prediction_text=f"🏠 Predicted Price: ₹ {round(price, 2)}"
        )

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)