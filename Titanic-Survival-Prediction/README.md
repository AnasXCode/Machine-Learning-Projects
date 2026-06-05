# 🚢 Titanic Survival Predictor

An interactive machine learning web application that predicts whether a Titanic passenger would have survived — built with Python, Scikit-learn, and Streamlit.

---

## 📌 Project Overview

This project is a complete end-to-end machine learning pipeline built on the classic [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic). It includes data cleaning, feature engineering, model training, and a fully interactive dashboard where you can enter any passenger's details and instantly get a survival prediction along with a detailed explanation of the reasons behind it.

> **Model Accuracy: 82.68%** on the validation set using a Random Forest Classifier.

---

## 🖥️ Dashboard Preview

The app has two pages:

| Page | Description  |
|------|--------------|
| 📊 Dataset Overview | View dataset stats, survival charts, class distribution, and missing value report |
| 🔍 Predict Survival | Enter passenger details and get a prediction with confidence score and reason summary |
### 📊 Interactive Analytics Interface
<img width="959" height="412" alt="Screenshot 2026-06-05 121014" src="https://github.com/user-attachments/assets/6f657ad9-0bb6-44b7-aad3-7d0e41d114ec" />
<img width="959" height="410" alt="image" src="https://github.com/user-attachments/assets/569dcdbd-f76b-47a7-9262-ae11ea82a19d" />
<img width="948" height="410" alt="image" src="https://github.com/user-attachments/assets/987e8df5-63d8-4f58-b47d-5ad210a19031" />

---

## 📁 Project Structure

```
titanic-survival-predictor/
│
├── app.py                   # Main Streamlit dashboard application
├── titanic_project.ipynb    # Jupyter Notebook (data exploration + model training)
├── train.csv                # Training dataset (891 passengers)
├── test.csv                 # Test dataset (418 passengers)
├── gender_submission.csv    # Sample submission file
└── README.md                # Project documentation (this file)
```

---

## 📊 Dataset

The dataset is from the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data).

| File                    | Rows        | Description |
|-------------------------|-------------|-------------|
| `train.csv`             | 891         | Labelled data used for training (includes `Survived` column) |
| `test.csv`              | 418         | Unlabelled data used for final prediction submission |
| `gender_submission.csv` | 418         | Example submission format |

### Feature Descriptions

| Column        | Type             | Description |
|---------------|------------------|-------------|
| `PassengerId` | Integer          | Unique ID for each passenger |
| `Survived`    | Binary (0/1)     | Target — 0 = Did Not Survive, 1 = Survived |
| `Pclass`      | Integer (1/2/3)  | Ticket class — 1st, 2nd, or 3rd |
| `Name`        | String           | Full name of the passenger |
| `Sex`         | String           | Gender — male or female |
| `Age`         | Float            | Age in years |
| `SibSp`       | Integer          | Number of siblings / spouses aboard |
| `Parch`       | Integer          | Number of parents / children aboard |
| `Ticket`      | String           | Ticket number |
| `Fare`        | Float            | Passenger fare in British pounds (£) |
| `Cabin`       | String           | Cabin number (mostly missing) |
| `Embarked`    | String           | Port of embarkation — S = Southampton, C = Cherbourg, Q = Queenstown |

### Key Dataset Statistics

- **Total Passengers (Train):** 891
- **Overall Survival Rate:** 38.4%
- **Female Survival Rate:** 74.2%
- **Male Survival Rate:** 18.9%
- **1st Class Survival Rate:** 63.0%
- **2nd Class Survival Rate:** 47.3%
- **3rd Class Survival Rate:** 24.2%

---

## ⚙️ How It Works

### 1. Data Preprocessing

- **Missing `Age` values** → filled with the median age (28 years)
- **Missing `Embarked` values** → filled with the most frequent port (`S` = Southampton)
- **`Cabin` column** → dropped (77% missing values, not reliable)
- **`Name`, `Ticket`, `PassengerId`** → dropped (not useful for prediction)
- **`Sex` encoding** → `male = 0`, `female = 1`
- **`Embarked` encoding** → `S = 0`, `C = 1`, `Q = 2`

### 2. Model Training

- **Algorithm:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Train/Validation Split:** 80% training, 20% validation (`random_state=42`)
- **Validation Accuracy:** **82.68%**

### 3. Feature Importance

The model ranks features by how much they influence the prediction:

| Feature | Importance |
|---------|----------- |
| Fare    | 27.1%      |
| Sex     | 26.1%      |
| Age     | 26.0%      |
| Pclass  | 8.8%       |
| SibSp   | 4.6%       |
| Parch   | 4.0%       |
| Embarked| 3.4%       |

### 4. Prediction Reasons

After every prediction, the dashboard shows two sections:
- ✅ **Factors That Helped Survival** — why the model leaned toward survival
- ❌ **Factors That Reduced Survival** — why the model leaned against survival

Each reason is based on real historical statistics from the training data, covering sex, class, age, fare, family size, and embarkation port.

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.7 or higher installed.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/titanic-survival-predictor.git
cd titanic-survival-predictor
```

### 2. Install Required Libraries

```bash
pip install streamlit pandas scikit-learn
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will automatically open in your browser at:
```
http://localhost:8501
```

> ⚠️ Make sure `train.csv` is in the **same folder** as `app.py` before running.

---

## 📦 Requirements

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and preprocessing |
| `scikit-learn` | Machine learning model (Random Forest) |
| `streamlit` | Interactive web dashboard |

Install all at once:

```bash
pip install pandas scikit-learn streamlit
```

---

## 🔍 How to Use the Prediction Page

1. Open the app and go to **🔍 Predict Survival** from the sidebar
2. Fill in the passenger details:
   - Select Passenger Class (1st / 2nd / 3rd)
   - Select Sex (Male / Female)
   - Set Age using the slider
   - Enter Fare paid (£)
   - Set number of Siblings/Spouses aboard
   - Set number of Parents/Children aboard
   - Select Port of Embarkation
3. Click **🔮 Predict Now**
4. The app will show:
   - ✅ Survived or ❌ Did Not Survive
   - Survival probability percentage
   - A visual progress bar showing confidence
   - A full reasons breakdown (positive and negative factors)
   - A passenger summary table

---

## 📓 Jupyter Notebook

The `titanic_project.ipynb` notebook walks through the full process step by step:

1. **Cell 1** — Load the data with `pandas` and display the first 5 rows
2. **Cell 2** — Check missing values in each column
3. **Cell 3** — Clean data: fill missing values, drop unused columns, encode categorical features
4. **Cell 4** — Train the Random Forest model and evaluate accuracy on the validation set

---

## 🧠 What is a Random Forest?

A Random Forest is an ensemble machine learning algorithm that builds many decision trees during training and combines their results to make more accurate and stable predictions. It is particularly well-suited for tabular datasets like Titanic because:

- It handles both numerical and categorical (encoded) data well
- It is resistant to overfitting compared to a single decision tree
- It provides feature importance scores showing which inputs matter most

---

## 📈 Model Performance

| Metric              | Value                    |
|---------------------|--------------------------|
| Validation Accuracy | **82.68%**               |
| Algorithm           | Random Forest Classifier |
| Train Size          | 712 passengers (80%)     |
| Validation Size     | 179 passengers (20%)     |

---

## 🛠️ Possible Improvements

- Tune hyperparameters (`n_estimators`, `max_depth`) using GridSearchCV
- Engineer new features such as `FamilySize = SibSp + Parch + 1` or title extracted from `Name`
- Use cabin deck (extracted from `Cabin`) as a feature where available
- Try other algorithms like XGBoost, LightGBM, or a Voting Classifier
- Add SHAP value visualizations for more precise model explainability

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Dataset provided by [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- Built with [Streamlit](https://streamlit.io/), [Scikit-learn](https://scikit-learn.org/), and [Pandas](https://pandas.pydata.org/)
