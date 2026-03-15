import os
import json
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# create output directory
os.makedirs("output", exist_ok=True)

# load dataset
df = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# predictions
pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MSE:", mse)
print("R2:", r2)

# save model
with open("output/model.pkl", "wb") as f:
    pickle.dump(model, f)

# save results
results = {
    "MSE": mse,
    "R2": r2
}

with open("output/results.json", "w") as f:
    json.dump(results, f)

# write GitHub summary
import os

if "GITHUB_STEP_SUMMARY" in os.environ:
    summary = f"""
## ML Experiment Results

Name: Tanishq Singh  
Roll No: bcs183  

MSE: {mse}
R2 Score: {r2}
"""
    with open(os.environ["GITHUB_STEP_SUMMARY"], "w") as f:
        f.write(summary)
