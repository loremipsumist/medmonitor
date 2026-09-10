from datetime import date
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st

st.set_page_config(
    page_title="Medicine Quality Prediction System", layout="wide"
)


# Train model on-the-fly and cache it in memory
@st.cache_resource
def get_trained_model_and_dataset():
    np.random.seed(42)
    n = 2000
    categories = [
        "Antibiotic",
        "Analgesic",
        "Antiviral",
        "Antihistamine",
        "Antifungal",
    ]
    dosage_forms = ["Tablet", "Syrup", "Injection", "Cream", "Capsule"]
    manufacturers = ["Pfizer", "Novartis", "Cipla", "SunPharma", "GSK", "Pluxe"]

    df = pd.DataFrame(
        {
            "category": np.random.choice(categories, n),
            "dosage_form": np.random.choice(dosage_forms, n),
            "manufacturer": np.random.choice(manufacturers, n),
            "strength_mg": np.random.choice([100, 250, 500, 650, 1000], n),
            "temp_celsius": np.random.normal(loc=24, scale=6, size=n),
            "humidity_pct": np.random.normal(loc=55, scale=12, size=n),
            "days_to_expiry": np.random.randint(-60, 730, size=n),
        }
    )

    df["is_consumable"] = 1
    df.loc[df["days_to_expiry"] <= 0, "is_consumable"] = 0
    df.loc[
        (df["temp_celsius"] > 32) | (df["temp_celsius"] < 2), "is_consumable"
    ] = 0
    df.loc[df["humidity_pct"] > 75, "is_consumable"] = 0

    X = df.drop(columns=["is_consumable"])
    y = df["is_consumable"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["category", "dosage_form", "manufacturer"],
            ),
            (
                "num",
                StandardScaler(),
                ["strength_mg", "temp_celsius", "humidity_pct", "days_to_expiry"],
            ),
        ]
    )

    pipeline = Pipeline(
        [
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    pipeline.fit(X, y)
    return pipeline, df


model, df = get_trained_model_and_dataset()

# Login State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown(
        "<h2 style='text-align: center;'>Login Page</h2>", unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        user = st.text_input("Username", value="admin")
        pwd = st.text_input("Password", type="password", value="admin")
        if st.button("Login", use_container_width=True):
            if user == "admin" and pwd == "admin":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials.")
else:
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio(
        "Menu",
        ["Welcome Page", "View Dataset", "Predict Medicine", "Dashboard"],
    )

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    if choice == "Welcome Page":
        st.title("Medicine Prediction System")
        st.info(
            "An intelligent healthcare application utilizing Machine Learning to predict medicine quality and consumable status."
        )

    elif choice == "View Dataset":
        st.title("Trained Medicine Dataset")
        st.write(f"Total entries: {len(df)}")
        st.dataframe(df, use_container_width=True)

    elif choice == "Predict Medicine":
        st.title("Medicine Quality Prediction")
        with st.form("input_form"):
            c1, c2 = st.columns(2)
            with c1:
                med_name = st.text_input("Medicine Name", value="Amoxazole")
                category = st.selectbox(
                    "Category",
                    [
                        "Antibiotic",
                        "Analgesic",
                        "Antiviral",
                        "Antihistamine",
                        "Antifungal",
                    ],
                )
                dosage = st.selectbox(
                    "Dosage Form",
                    ["Tablet", "Syrup", "Injection", "Cream", "Capsule"],
                )
                mfg = st.selectbox(
                    "Manufacturer",
                    ["Pfizer", "Novartis", "Cipla", "SunPharma", "GSK", "Pluxe"],
                )
            with c2:
                strength = st.number_input(
                    "Strength (mg)", min_value=10, max_value=2000, value=500
                )
                exp = st.date_input(
                    "Expiry Date",
                    value=date.today().replace(year=date.today().year + 1),
                )
                temp = st.slider(
                    "Storage Temperature (°C)", -5.0, 50.0, value=22.0
                )
                humidity = st.slider("Storage Humidity (%)", 10.0, 100.0, value=50.0)
            btn = st.form_submit_button(
                "Predict Quality", use_container_width=True
            )

        if btn:
            days_left = (exp - date.today()).days
            row = pd.DataFrame(
                [
                    {
                        "category": category,
                        "dosage_form": dosage,
                        "manufacturer": mfg,
                        "strength_mg": strength,
                        "temp_celsius": temp,
                        "humidity_pct": humidity,
                        "days_to_expiry": days_left,
                    }
                ]
            )
            res = model.predict(row)[0]
            st.markdown("---")
            st.subheader("Prediction Result")
            if res == 1:
                st.success(
                    "Predicted Quality Status: Good | Consumable Status: Consumable"
                )
            else:
                st.error(
                    "Predicted Quality Status: Poor | Consumable Status: Non-Consumable"
                )

    elif choice == "Dashboard":
        st.title("Quality Monitoring Dashboard")
        c1, c2 = st.columns(2)
        c1.metric("Total Batches", len(df))
        c2.metric(
            "Consumable Ratio", f"{(df['is_consumable'].mean() * 100):.1f}%"
        )
        st.scatter_chart(
            df, x="temp_celsius", y="humidity_pct", color="is_consumable"
        )
