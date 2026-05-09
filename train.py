import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import pickle

df = pd.read_csv("House Price Prediction Dataset.csv").dropna()
df = df.drop("Id", axis=1)

X = df.drop("Price", axis=1)
y = df["Price"]

categorical = ["Location", "Condition", "Garage"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder="passthrough")

model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6
)

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

pipeline.fit(X, y)

pickle.dump(pipeline, open("model.pkl", "wb"))

print("Model trained correctly")