from flask import Flask, request, render_template
import pickle
import pandas as pd
from database import init_db
import sqlite3

init_db()
app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        area = float(data['area'])
        bedrooms = float(data['bedrooms'])
        bathrooms = float(data['bathrooms'])
        floors = float(data['floors'])
        yearbuilt = float(data['yearbuilt'])

        input_data = pd.DataFrame([{
            'Id': 1,
            'Area': area,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Floors': floors,
            'YearBuilt': yearbuilt
        }])

        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=columns, fill_value=0)

        price = model.predict(input_data)[0]

        conn = sqlite3.connect("predictions.db")
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO predictions (area, bedrooms, bathrooms, floors, yearbuilt, predicted_price)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (area, bedrooms, bathrooms, floors, yearbuilt, price))

        conn.commit()
        conn.close()

        return render_template("index.html", prediction_text=f"🏠 Predicted House Price: ₹ {round(price, 2)}")

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)