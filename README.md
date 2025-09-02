✈️ Flight Fare Predictor – Model Comparison Dashboard

A machine learning project that compares multiple regression models for predicting flight ticket prices.
The app is built with Streamlit and shows model metrics (RMSE, R²) and feature importances to help evaluate performance.

📌 Features

📊 Dashboard UI built with Streamlit

✅ Supports multiple trained models:

Linear Regression

Ridge & Lasso Regression

Decision Tree

Random Forest

Gradient Boosting

XGBoost

📈 Model comparison table (RMSE & R²)

🔍 Feature importance visualization for tree-based models

🗂️ Uses saved model artifacts (.pkl) for reproducibility

🕑 Backdated initial commit (02 Sep 2025) for timeline consistency

🏗️ Project Structure
flight-fare-predictor/
│
├── app.py                     # Streamlit dashboard
├── models/                    # Saved models & artifacts
│   ├── gbr_reg.pkl
│   ├── rf_reg.pkl
│   ├── xgb_reg.pkl
│   ├── dt_reg.pkl
│   ├── linear_reg.pkl
│   ├── ridge_reg.pkl
│   ├── lasso_reg.pkl
│   ├── ohe_encoder.pkl
│   ├── feature_columns.pkl
│   └── model_metrics.json
│
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
└── .gitignore

⚙️ Installation

Clone the repo and install dependencies:

git clone https://github.com/<your-username>/flight-fare-predictor.git
cd flight-fare-predictor
pip install -r requirements.txt

▶️ Run the App
streamlit run app.py


Then open the app in your browser → http://localhost:8501

📊 Sample Dashboard

Model Comparison Table:

Model	RMSE	R²
GradientBoosting	2300.45	0.86
XGBoost	2405.21	0.85
RandomForest	2501.77	0.84
DecisionTree	3200.91	0.76
LinearRegression	4200.34	0.65

(Values above are illustrative — actual results depend on training data.)

🧠 Models & Training

All models were trained on a flight fare dataset (with features like Airline, Source, Destination, Date_of_Journey, Duration, Stops, etc.).

Feature engineering included:

Extracting Day, Month, Year from travel date

Parsing departure & arrival times

Handling total stops & duration

One-hot encoding categorical features

Models were evaluated using RMSE (Root Mean Squared Error) and R² score.

🚀 Future Work

Add interactive fare prediction form (user enters flight details → app predicts price)

Integrate APIs (Google Maps, Amadeus) to auto-derive distance & duration

Improve UI with charts & comparison plots

🙌 Acknowledgements

Streamlit

scikit-learn

XGBoost

