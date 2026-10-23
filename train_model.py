import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "attendance": [50, 55, 60, 65, 70, 75, 80, 90],
    "marks": [30, 35, 40, 48, 55, 65, 75, 90]
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance"]]
y = df["marks"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "student_result_model.pkl")

print("Model trained successfully!")
