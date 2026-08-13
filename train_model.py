import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("dataset/customer_churn.csv")

encoders = {}

for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        encoders[column] = le

X = df.drop("Churn", axis=1)
y = df["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train,y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test,pred)

print("Accuracy:",accuracy)

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Stay','Leave'],
    yticklabels=['Stay','Leave']
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.show()

plt.figure(figsize=(5,5))

plt.bar(
    ["Accuracy"],
    [accuracy * 100]
)

plt.ylim(0,100)

plt.title("Model Accuracy")
plt.ylabel("Percentage")

plt.savefig("accuracy_graph.png")
plt.show()

importance = model.feature_importances_

feature_names = X.columns

indices = np.argsort(importance)

plt.figure(figsize=(8,5))

plt.barh(
    range(len(indices)),
    importance[indices]
)

plt.yticks(
    range(len(indices)),
    [feature_names[i] for i in indices]
)

plt.title("Feature Importance")

plt.xlabel("Importance Score")

plt.savefig("feature_importance.png")
plt.show()

joblib.dump(model,"model/churn_model.pkl")
joblib.dump(scaler,"model/scaler.pkl")
joblib.dump(encoders,"model/encoders.pkl")

print("Model Saved Successfully")