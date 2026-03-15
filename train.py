import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os 

os.makedirs("output", exist_ok=True) 

# load dataset
df = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = LinearRegression()
model.fit(X_train, y_train)

# prediction
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

with open("output/results.json") as f:
    data = json.load(f)

summary = f"""
## ML Experiment Results

Name: Tanishq Singh  
Roll No: bcs183  

MSE: {data['MSE']}
R2 Score: {data['R2']}
"""

with open(os.environ['GITHUB_STEP_SUMMARY'], 'w') as f:
    f.write(summary)
