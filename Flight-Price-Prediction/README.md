# ✈️ Flight Price Prediction — Machine Learning Project

A complete end-to-end Machine Learning project that predicts Indian domestic flight ticket prices using regression algorithms. Built with Python, Scikit-learn, and XGBoost — with a fully functional **SkyFare** web app built on Streamlit.

---

## 🌐 Live Demo (Local)

![SkyFare App](screenshots/skyfare_app.png)

> **SkyFare** — Instant, data-driven flight price estimates across India, powered by machine learning trained on millions of fares.

**How to run the web app:**
```bash
streamlit run app.py
```
Then open: `http://localhost:8501`

---

## 📦 Download Trained Model

> The trained model file is too large for GitHub (934 MB).
> Download it directly from Kaggle:

| File | Description | Link |
|---|---|---|
| `flight_price_rf_model.pkl` | Trained Random Forest Model | [⬇️ Download from Kaggle](https://www.kaggle.com/datasets/anasahmed565/flight-price-prediction-model) |
| `model_features.pkl` | Feature columns list | [⬇️ Download from Kaggle](https://www.kaggle.com/datasets/anasahmed565/flight-price-prediction-model) |

**After downloading**, place both files in the project root folder:
```
Flight-Price-Prediction/
├── flight_price_rf_model.pkl   ← download from Kaggle
├── model_features.pkl          ← download from Kaggle
├── app.py
├── Flight_Price_Prediction.ipynb
```

---

## 📌 Project Overview

Flight prices are highly dynamic and depend on many factors like airline, travel class, number of stops, days left for booking, and route. This project uses a real-world dataset scraped from EaseMyTrip to train multiple ML models and predict ticket prices with high accuracy.

**ML Type:** Supervised Learning — Regression  
**Best Model:** Random Forest Regressor  
**Best Accuracy:** 98.48% (R² Score)  
**Web App:** Streamlit (SkyFare UI)

---

## 📂 Dataset

- **Source:** [Kaggle — Flight Price Prediction by Shubham Bathwal](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
- **Size:** 300,153 rows × 12 columns
- **Classes:** Economy & Business

### Features

| Column | Description |
|---|---|
| `airline` | Name of the airline |
| `flight` | Flight number |
| `source_city` | City of departure |
| `departure_time` | Time of departure (Morning, Evening, etc.) |
| `stops` | Number of stops (zero, one, two_or_more) |
| `arrival_time` | Time of arrival |
| `destination_city` | City of arrival |
| `class` | Travel class (Economy / Business) |
| `duration` | Flight duration in hours |
| `days_left` | Days between booking date and travel date |
| `price` | 🎯 Target variable — ticket price in INR |

---

## 🔧 Tech Stack

- Python 3.14
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Joblib (for model saving)
- **Streamlit** (for web app UI)

---

## 🚀 Project Pipeline

```
Raw CSV Data
    │
    ▼
Data Cleaning
(Drop 'flight' & 'Unnamed: 0' columns)
    │
    ▼
Feature Engineering
(Map 'stops' → 0, 1, 2)
    │
    ▼
One-Hot Encoding
(Convert text columns → 29 binary features)
    │
    ▼
Train-Test Split
(80% Train / 20% Test)
    │
    ▼
Train & Compare 6 ML Models
    │
    ▼
Best Model Selected → Random Forest (98.48%)
    │
    ▼
Model Saved as .pkl file
    │
    ▼
Streamlit Web App (SkyFare)
```

---

## 📊 Model Comparison Results

| Model | R² Score (Accuracy) | MAE (Error in INR) |
|---|---|---|
| Linear Regression | 90.99% | 4500.71 |
| Ridge Regression | 90.99% | 4500.70 |
| Lasso Regression | 90.99% | 4500.54 |
| Decision Tree | 97.55% | 1182.29 |
| **Random Forest** | **98.48%** | **1086.16** |
| XGBoost | 96.76% | 2348.03 |

✅ **Winner: Random Forest Regressor**

---

## 🔍 Overfitting Verification

To make sure the model genuinely learned patterns and didn't just memorize the data:

| Check | Score |
|---|---|
| Training Accuracy | 99.75% |
| Testing Accuracy | 98.48% |
| Variance (Difference) | **1.27%** ✅ |

A variance of less than 2% confirms the model is **NOT overfitted** and is reliable for real-world use.

---

## 💡 Key Insights from EDA

- **Vistara and Air India** are the most expensive airlines on average.
- **Business class** tickets cost significantly more (avg ~50,000 INR) than Economy (avg ~6,000–10,000 INR).
- **Days left** has a major impact on price — tickets booked 2 days before departure are significantly more expensive than those booked 40 days in advance (dynamic pricing).

---

## 🧪 Sample Predictions

**Scenario 1 — 10 days before departure:**
```
Airline:     Vistara
Route:       Delhi → Mumbai
Class:       Business | Stops: 1
Duration:    5.5 hrs | Days Left: 10

💰 Predicted Price: 47,026.70 INR
```

**Scenario 2 — 2 days before departure (last minute):**
```
Airline:     Vistara
Route:       Delhi → Mumbai
Class:       Business | Stops: 1
Duration:    5.5 hrs | Days Left: 2

💰 Predicted Price: 61,846.50 INR
```

**Scenario 3 — From SkyFare Web App:**
```
Airline:     Air India
Route:       Bangalore → Hyderabad
Class:       Business | Stops: 0
Duration:    6.0 hrs | Days Left: 15

💰 Predicted Price: ₹56,476 INR
```

The model correctly captures real-world **dynamic pricing** behavior.

---

## ⚙️ How to Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/AnasXCode/Machine-Learning-Projects.git
cd Machine-Learning-Projects/Flight-Price-Prediction
```

**2. Install required libraries:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost joblib streamlit
```

**3. Download model files from Kaggle:**

👉 [https://www.kaggle.com/datasets/anasahmed565/flight-price-prediction-model](https://www.kaggle.com/datasets/anasahmed565/flight-price-prediction-model)

Place both `.pkl` files in the project folder.

**4. Run the Streamlit web app:**
```bash
streamlit run app.py
```

**5. Or open the training notebook:**
```bash
jupyter notebook Flight_Price_Prediction.ipynb
```

---

## 📁 Project Structure

```
Flight-Price-Prediction/
│
├── app.py                         # SkyFare Streamlit web app
├── Clean_Dataset.csv              # Cleaned dataset used for training
├── Flight_Price_Prediction.ipynb  # Main Jupyter Notebook (full pipeline)
├── flight_price_rf_model.pkl      # Download from Kaggle (see above)
├── model_features.pkl             # Download from Kaggle (see above)
└── README.md                      # This file
```

---

## 🔮 Future Improvements

- Hyperparameter tuning with `RandomizedSearchCV` to push accuracy closer to 99%
- Feature engineering: combining `source_city` + `destination_city` into a `route` column
- Remove outliers (top 1% extreme prices) to reduce remaining error
- Deploy SkyFare app on **Streamlit Cloud** for public access
- Build a REST API using **FastAPI** for mobile app integration

---

## 👨‍💻 Author

**AnasXCode** — Built as part of an AI/ML learning roadmap.  
Completed end-to-end: data cleaning, EDA, preprocessing, model benchmarking, validation, deployment-ready model export, and full Streamlit web app (SkyFare).
