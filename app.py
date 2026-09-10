from datetime import date
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Online Testing & Monitoring Quality Medicines", layout="wide"
)


# Authentication Simulation (Fig. 3)
def check_auth():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("<h2 style='text-align: center;'>Login Page</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            username = st.text_input("Username", value="admin")
            password = st.text_input("Password", type="password", value="admin")
            if st.button("Login", use_container_width=True):
                if username == "admin" and password == "admin":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        return False
    return True


if check_auth():
    # Load Model
    @st.cache_resource
    def load_model():
        return joblib.load("medicine_quality_pipeline.pkl")

    try:
        model = load_model()
    except FileNotFoundError:
        st.error(
            "Model artifact not found. Please execute `python train_model.py` first."
        )
        st.stop()

    # Navigation Sidebar
    st.sidebar.title("System Navigation")
    menu = st.sidebar.radio(
        "Menu", ["Welcome Page", "View Dataset", "Predict Medicine", "Dashboard"]
    )

    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    # 1. Welcome Page (Fig. 2)
    if menu == "Welcome Page":
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

    # 2. View Dataset Page (Fig. 4)
    elif menu == "View Dataset":
        st.title("Medicine Quality Dataset")
        try:
            df = pd.read_csv("medicine_training_data.csv")
            st.write(f"Total Records: {len(df)}")
            st.dataframe(df, use_container_width=True)
        except FileNotFoundError:
            st.warning("Training dataset CSV not found.")

    # 3. Predict Quality Page (Fig. 5 & Fig. 6)
    elif menu == "Predict Medicine":
        st.title("Medicine Quality Prediction")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
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
                dosage_form = st.selectbox(
                    "Dosage Form",
                    ["Tablet", "Syrup", "Injection", "Cream", "Capsule"],
                )
                manufacturer = st.selectbox(
                    "Manufacturer",
                    ["Pfizer", "Novartis", "Cipla", "SunPharma", "GSK", "Pluxe"],
                )

            with col2:
                strength_mg = st.number_input(
                    "Strength (mg)", min_value=10, max_value=2000, value=500
                )
                expiry_date = st.date_input(
                    "Expiry Date", value=date.today().replace(year=date.today().year + 1)
                )
                temp = st.slider(
                    "Storage Temperature (°C)",
                    min_value=-5.0,
                    max_value=50.0,
                    value=22.0,
                )
                humidity = st.slider(
                    "Storage Humidity (%)", min_value=10.0, max_value=100.0, value=50.0
                )

            submit = st.form_submit_button("Predict Quality Status", use_container_width=True)

        if submit:
            days_to_expiry = (expiry_date - date.today()).days

            input_data = pd.DataFrame(
                [
                    {
                        "category": category,
                        "dosage_form": dosage_form,
                        "manufacturer": manufacturer,
                        "strength_mg": strength_mg,
                        "temp_celsius": temp,
                        "humidity_pct": humidity,
                        "days_to_expiry": days_to_expiry,
                    }
                ]
            )

            prediction = model.predict(input_data)[0]
            confidence = model.predict_proba(input_data)[0][prediction] * 100

            st.markdown("---")
            st.subheader("Prediction Result")
            if prediction == 1:
                st.success(
                    f"**Predicted Quality Status:** Good\n\n**Consumable Status:** Consumable (Confidence: {confidence:.1f}%)"
                )
            else:
                st.error(
                    f"**Predicted Quality Status:** Poor / Compromised\n\n**Consumable Status:** Non-Consumable (Confidence: {confidence:.1f}%)"
                )

    # 4. Monitoring Dashboard (Fig. 1 System Layer)
    elif menu == "Dashboard":
        st.title("Quality Monitoring Dashboard")
        df = pd.read_csv("medicine_training_data.csv")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Batches Analyzed", len(df))
        col2.metric(
            "Consumable Rate",
            f"{(df['is_consumable'].mean() * 100):.1f}%",
        )
        col3.metric("Defect/Expired Rate", f"{((1 - df['is_consumable'].mean()) * 100):.1f}%")

        st.subheader("Temperature vs. Consumability")
        st.scatter_chart(df, x="temp_celsius", y="humidity_pct", color="is_consumable")
