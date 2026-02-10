import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Sample dataset
data = {
    "area": [800, 1000, 1200, 1500, 1800, 2000],
    "bedrooms": [1, 2, 2, 3, 3, 4],
    "bathrooms": [1, 1, 2, 2, 3, 3],
    "price": [40, 50, 65, 80, 100, 120]  # in lakhs
}

df = pd.DataFrame(data)

X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "house_price_model.pkl")
print("Model saved!")
