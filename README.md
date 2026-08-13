# Customer Churn Prediction

A machine learning web application that predicts whether a customer is likely to leave (churn) or stay, based on their account details. Built with **scikit-learn** for the model and **Flask** for the web interface.

## Overview

The app trains a Random Forest classifier on customer data (tenure, charges, contract type, internet service) and serves predictions through a simple web form. Enter a customer's details, and the app tells you whether they're likely to stay or leave.

## Project Structure

```
Customer-Churn-Prediction/
├── app.py                     # Flask application (routes + prediction logic)
├── train_model.py             # Model training script
├── requirements.txt           # Python dependencies
├── dataset/
│   └── customer_churn.csv     # Training data
├── model/
│   ├── churn_model.pkl        # Trained Random Forest model
│   ├── scaler.pkl             # Fitted StandardScaler
│   └── encoders.pkl           # Fitted LabelEncoders for categorical columns
├── templates/
│   ├── index.html             # Input form
│   ├── result.html            # Prediction result page
│   └── dashboard.html         # Dashboard view
├── static/
│   └── css/
│       └── style.css          # App styling
├── confusion_matrix.png       # Model evaluation: confusion matrix
├── accuracy_graph.png         # Model evaluation: accuracy
└── feature_importance.png     # Model evaluation: feature importance
```

## Features

- Predicts customer churn using a trained Random Forest classifier
- Simple, clean web form for entering customer details
- Visual evaluation of model performance (confusion matrix, accuracy, feature importance)
- Standardized inputs via `StandardScaler` for consistent predictions

## Input Fields

| Field | Description |
|---|---|
| Tenure | Number of months the customer has stayed |
| Monthly Charges | Customer's monthly bill amount |
| Total Charges | Customer's total charges to date |
| Contract | Month-to-Month, One Year, or Two Year |
| Internet Service | DSL, Fiber, or No Internet |

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Customer-Churn-Prediction
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model (optional)

A pre-trained model is already included in `model/`. To retrain it on the dataset:

```bash
python train_model.py
```

This will:
- Load and preprocess `dataset/customer_churn.csv`
- Train a `RandomForestClassifier`
- Save the model, scaler, and encoders to `model/`
- Generate evaluation charts (`confusion_matrix.png`, `accuracy_graph.png`, `feature_importance.png`)

### Running the App

```bash
python app.py
```

Then open your browser to `http://127.0.0.1:5000`.

## How It Works

1. The user submits customer details through the web form (`index.html`).
2. `app.py` collects the inputs and scales them using the saved `scaler.pkl`.
3. The scaled features are passed to the trained model (`churn_model.pkl`) for prediction.
4. The result — **"Customer Will Leave"** or **"Customer Will Stay"** — is displayed on `result.html`.

## Tech Stack

- **Python**
- **Flask** — web framework
- **scikit-learn** — model training (Random Forest, StandardScaler, LabelEncoder)
- **pandas / numpy** — data handling
- **matplotlib / seaborn** — evaluation visualizations
- **joblib** — model persistence

## Model Evaluation

Evaluation artifacts are generated during training and saved as images:
- `confusion_matrix.png` — classification performance breakdown
- `accuracy_graph.png` — overall model accuracy
- `feature_importance.png` — which features most influence predictions

## Future Improvements

- Expand the dataset for better generalization
- Add more customer features (e.g. payment method, tech support, demographics)
- Add input validation on the form
- Deploy the app to a hosting platform (e.g. Render, Heroku, AWS)

## License

This project is available for personal and educational use. Add a license of your choice if distributing publicly.
