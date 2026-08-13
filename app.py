from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    tenure = float(request.form['tenure'])
    monthly = float(request.form['monthly'])
    total = float(request.form['total'])
    contract = int(request.form['contract'])
    internet = int(request.form['internet'])

    features = np.array([
        [
            tenure,
            monthly,
            total,
            contract,
            internet
        ]
    ])

    features = scaler.transform(features)

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Customer Will Leave"
    else:
        result = "Customer Will Stay"

    return render_template(
        'result.html',
        prediction=result
    )

if __name__ == "__main__":
    app.run(debug=True)