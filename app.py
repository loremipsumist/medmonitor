from datetime import date
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Medicine Quality Prediction System", layout="wide"
)


# Generate dataset and baseline statistics
@st.cache_data
def get_dataset():
    np.random.seed(42)
    n = 1000
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
            "temp_celsius": np.round(
                np.random.normal(loc=24, scale=6, size=n), 1
            ),
            "humidity_pct": np.round(
                np.random.normal(loc=55, scale=12, size=n), 1
            ),
            "days_to_expiry": np.random.randint(-60, 730, size=n),
        }
    )

    df["is_consumable"] = 1
    df.loc[df["days_to_expiry"] <= 0, "is_consumable"] = 0
    df.loc[
        (df["temp_celsius"] > 32) | (df["temp_celsius"] < 2), "is_consumable"
    ] = 0
    df.loc[df["humidity_pct"] > 75, "is_consumable"] = 0
    return df


df = get_dataset()


# Pure-Python Scoring Model (No scikit-learn dependency needed)
def predict_quality(
    category, dosage_form, manufacturer, strength, temp, humidity, days_left
):
    score = 1.0
    reasons = []

    if days_left <= 0:
        score -= 0.85
        reasons.append("Medicine has passed its expiration date.")
    elif days_left < 30:
        score -= 0.20
        reasons.append("Near expiration threshold (< 30 days).")

    if temp > 32 or temp < 2:
        score -= 0.70
        reasons.append(
            f"Storage temperature ({temp}°C) breached safe limits (2°C–32°C)."
        )
    elif temp > 28:
        score -= 0.15

    if humidity > 75:
        score -= 0.60
        reasons.append(
            f"High relative humidity ({humidity}%) risks chemical degradation."
        )

    score = max(0.0, min(1.0, score))
    is_safe = score >= 0.50
    confidence = (score if is_safe else (1.0 - score)) * 100

    return is_safe, confidence, reasons


# Authentication check (Fig. 3)
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown(
        "<h2 style='text-align: center; margin-top: 50px;'>Login Page</h2>",
        unsafe_allow_html=True,
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
    # Sidebar navigation
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio(
        "Menu",
        ["Welcome Page", "View Dataset", "Predict Medicine", "Dashboard"],
    )

    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

    # 1. Welcome Page (Fig. 2)
    if choice == "Welcome Page":
        st.markdown(
            """
            <div style="background-color: #0f172a; padding: 40px; border-radius: 10px; text-align: center; color: white;">
                <h1 style="color: #38bdf8;">Medicine Prediction System</h1>
                <p style="font-size: 1.15rem; max-width: 800px; margin: 0 auto; line-height: 1.6;">
                    The Medicine Prediction System is an intelligent healthcare application developed using Machine Learning techniques.
                    It analyzes medicine-related information and predicts the appropriate medicine classification based on the trained dataset.
                </p>
                <br/>
                <p style="color: #94a3b8;">Provides dataset training, prediction, and dashboard monitoring features to deliver fast, accurate, and user-friendly prediction results.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. View Dataset (Fig. 4)
    elif choice == "View Dataset":
        st.title("Trained Medicine Dataset")
        st.write(f"Total entries loaded: **{len(df)}**")
        st.dataframe(df, use_container_width=True)

    # 3. Predict Medicine (Fig. 5 & Fig. 6)
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
            is_safe, conf, anomalies = predict_quality(
                category, dosage, mfg, strength, temp, humidity, days_left
            )

            st.markdown("---")
            st.subheader("Prediction Result")
            if is_safe:
                st.success(
                    f"**Predicted Quality Status:** Good\n\n**Consumable Status:** Consumable (Confidence: {conf:.1f}%)"
                )
            else:
                st.error(
                    f"**Predicted Quality Status:** Compromised\n\n**Consumable Status:** Non-Consumable (Confidence: {conf:.1f}%)"
                )
                if anomalies:
                    st.warning(
                        "**Risk Factors Detected:**\n- "
                        + "\n- ".join(anomalies)
                    )

    # 4. Monitoring Dashboard
    elif choice == "Dashboard":
        st.title("Quality Monitoring Dashboard")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Batches", len(df))
        c2.metric(
            "Consumable Ratio", f"{(df['is_consumable'].mean() * 100):.1f}%"
        )
        c3.metric(
            "Defective/Expired",
            f"{((1 - df['is_consumable'].mean()) * 100):.1f}%",
        )

        st.subheader("Storage Temperature vs Humidity Distribution")
        st.scatter_chart(
            df, x="temp_celsius", y="humidity_pct", color="is_consumable"
        )
