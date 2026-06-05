import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# ─────────────────────────────────────────────
#  Load & preprocess data (cached so it runs once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("train.csv")

    # Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("S")

    # Drop unused columns
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    # Encode categorical columns
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # Split features and target
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Validation accuracy
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    return model, acc, df

model, accuracy, df = load_and_train()

# ─────────────────────────────────────────────
#  Helper: build reason bullets from inputs
# ─────────────────────────────────────────────
def build_reasons(sex, pclass, age, fare, sibsp, parch, embarked, survived_prob):
    """
    Returns (positive_reasons, negative_reasons) as lists of strings.
    Logic is based on actual Titanic survival statistics from the training data.
    """
    positive = []
    negative = []

    # --- Sex (biggest factor after Fare) ---
    if sex == "female":
        positive.append("✅ Being female greatly increased survival chances — 74% of women on Titanic survived due to 'women and children first' policy.")
    else:
        negative.append("❌ Being male significantly reduced survival chances — only 19% of men on Titanic survived.")

    # --- Passenger Class ---
    if pclass == 1:
        positive.append("✅ 1st class passengers had the highest survival rate (63%) — they had easier access to lifeboats on upper decks.")
    elif pclass == 2:
        positive.append("✅ 2nd class passengers had a moderate survival rate (~47%) compared to 3rd class.")
    else:
        negative.append("❌ 3rd class passengers had the lowest survival rate (24%) — their cabins were deep in the ship, far from lifeboats.")

    # --- Age ---
    if age <= 10:
        positive.append("✅ Children under 10 were given priority in lifeboats — young age increased survival odds.")
    elif age <= 30:
        positive.append("✅ Being young (under 30) was a slight advantage — better physical ability to reach lifeboats.")
    elif age >= 60:
        negative.append("❌ Older passengers (60+) had lower survival rates — less physical mobility to reach lifeboats quickly.")

    # --- Fare ---
    if fare >= 50:
        positive.append(f"✅ High fare paid (£{fare:.1f}) indicates a better cabin location closer to the deck and lifeboats.")
    elif fare >= 15:
        positive.append(f"✅ Average fare (£{fare:.1f}) — neither a strong positive nor negative factor.")
    else:
        negative.append(f"❌ Low fare (£{fare:.1f}) suggests a lower-deck cabin, further from lifeboats and escape routes.")

    # --- Family size ---
    family_size = sibsp + parch
    if family_size == 0:
        negative.append("❌ Travelling alone — solo passengers had slightly lower survival rates than small families.")
    elif family_size <= 3:
        positive.append(f"✅ Small family size ({family_size} members) — small families were easier to keep together and evacuate.")
    else:
        negative.append(f"❌ Large family size ({family_size} members) — large families struggled to evacuate together, lowering survival odds.")

    # --- Embarkation port ---
    if embarked == "C":
        positive.append("✅ Boarded at Cherbourg (C) — Cherbourg passengers had the highest survival rate among all ports (~55%).")
    elif embarked == "Q":
        negative.append("❌ Boarded at Queenstown (Q) — mostly 3rd class Irish emigrants, associated with lower survival rates.")
    else:
        negative.append("❌ Boarded at Southampton (S) — the majority of passengers boarded here; survival rate was average (~34%).")

    return positive, negative


# ─────────────────────────────────────────────
#  Sidebar – navigation
# ─────────────────────────────────────────────
st.sidebar.title("🚢 Titanic Dashboard")
page = st.sidebar.radio(
    "Go to",
    ["📊 Dataset Overview", "🔍 Predict Survival"]
)

# ═══════════════════════════════════════════════════════════
#  PAGE 1 – Dataset Overview
# ═══════════════════════════════════════════════════════════
if page == "📊 Dataset Overview":
    st.title("📊 Titanic Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Passengers", len(df))
    col2.metric("Survived", int(df["Survived"].sum()))
    col3.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    st.divider()

    st.subheader("Sample Data (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Survival Count")
        survival_counts = df["Survived"].value_counts().rename({0: "Did Not Survive", 1: "Survived"})
        st.bar_chart(survival_counts)

    with col_b:
        st.subheader("Passengers by Class")
        pclass_counts = df["Pclass"].value_counts().sort_index().rename({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})
        st.bar_chart(pclass_counts)

    st.divider()

    st.subheader("Missing Values in Original Data")
    raw = pd.read_csv("train.csv")
    missing = raw.isnull().sum()
    missing = missing[missing > 0]
    st.dataframe(
        missing.reset_index().rename(columns={"index": "Column", 0: "Missing Count"}),
        use_container_width=True
    )

# ═══════════════════════════════════════════════════════════
#  PAGE 2 – Predict Survival
# ═══════════════════════════════════════════════════════════
elif page == "🔍 Predict Survival":
    st.title("🔍 Predict Passenger Survival")
    st.markdown("Enter a passenger's details below to predict whether they would have survived.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox(
            "Passenger Class (Pclass)",
            options=[1, 2, 3],
            format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} Class"
        )

        sex = st.selectbox(
            "Sex",
            options=["male", "female"]
        )

        age = st.slider(
            "Age",
            min_value=1,
            max_value=80,
            value=28,
            step=1
        )

        fare = st.number_input(
            "Fare Paid (£)",
            min_value=0.0,
            max_value=600.0,
            value=32.2,
            step=0.5
        )

    with col2:
        sibsp = st.slider(
            "Siblings / Spouses aboard (SibSp)",
            min_value=0,
            max_value=8,
            value=0
        )

        parch = st.slider(
            "Parents / Children aboard (Parch)",
            min_value=0,
            max_value=6,
            value=0
        )

        embarked = st.selectbox(
            "Port of Embarkation",
            options=["S", "C", "Q"],
            format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x]
        )

    st.divider()

    if st.button("🔮 Predict Now", use_container_width=True, type="primary"):
        # Encode inputs exactly as training preprocessing
        sex_encoded = 0 if sex == "male" else 1
        embarked_encoded = {"S": 0, "C": 1, "Q": 2}[embarked]

        input_data = pd.DataFrame([{
            "Pclass":   pclass,
            "Sex":      sex_encoded,
            "Age":      age,
            "SibSp":    sibsp,
            "Parch":    parch,
            "Fare":     fare,
            "Embarked": embarked_encoded
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        survived_prob = probability[1] * 100
        died_prob     = probability[0] * 100

        st.divider()

        # ── Main result banner ──
        if prediction == 1:
            st.success("✅ **This passenger would have SURVIVED!**")
        else:
            st.error("❌ **This passenger would NOT have survived.**")

        # ── Probability metrics ──
        st.markdown("### Prediction Confidence")
        col_p1, col_p2 = st.columns(2)
        col_p1.metric("Survival Probability",        f"{survived_prob:.1f}%")
        col_p2.metric("Did Not Survive Probability",  f"{died_prob:.1f}%")
        st.progress(int(survived_prob), text=f"Survival chance: {survived_prob:.1f}%")

        st.divider()

        # ── WHY section ──
        st.markdown("### 🧠 Why This Prediction? — Key Reasons")

        positive_reasons, negative_reasons = build_reasons(
            sex, pclass, age, fare, sibsp, parch, embarked, survived_prob
        )

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.markdown("#### Factors That Helped Survival")
            if positive_reasons:
                for r in positive_reasons:
                    st.info(r)
            else:
                st.info("No strong positive factors found for this passenger.")

        with col_r2:
            st.markdown("#### Factors That Reduced Survival")
            if negative_reasons:
                for r in negative_reasons:
                    st.warning(r)
            else:
                st.warning("No strong negative factors found for this passenger.")

        st.divider()

        # ── Passenger summary table ──
        st.markdown("### Passenger Summary")
        summary = {
            "Class":      f"{pclass}{'st' if pclass==1 else 'nd' if pclass==2 else 'rd'}",
            "Sex":        sex.capitalize(),
            "Age":        age,
            "Fare":       f"£{fare:.2f}",
            "SibSp":      sibsp,
            "Parch":      parch,
            "Embarked":   {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[embarked],
            "Prediction": "Survived ✅" if prediction == 1 else "Did Not Survive ❌"
        }
        st.table(pd.DataFrame(summary.items(), columns=["Feature", "Value"]))