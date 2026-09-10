import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Synthesize a domain-representative dataset matching the paper's parameters
np.random.seed(42)
n_samples = 2500

categories = ["Antibiotic", "Analgesic", "Antiviral", "Antihistamine", "Antifungal"]
dosage_forms = ["Tablet", "Syrup", "Injection", "Cream", "Capsule"]
manufacturers = ["Pfizer", "Novartis", "Cipla", "SunPharma", "GSK", "Pluxe"]

data = {
    "category": np.random.choice(categories, n_samples),
    "dosage_form": np.random.choice(dosage_forms, n_samples),
    "manufacturer": np.random.choice(manufacturers, n_samples),
    "strength_mg": np.random.choice([100, 250, 500, 650, 1000], n_samples),
    "temp_celsius": np.random.normal(loc=24, scale=6, size=n_samples),  # Storage temp
    "humidity_pct": np.random.normal(loc=55, scale=12, size=n_samples),  # Storage humidity
    "days_to_expiry": np.random.randint(-60, 730, size=n_samples),  # < 0 means expired
}

df = pd.DataFrame(data)

# Ground-truth logic: determine 'Consumable' (1) vs 'Non-Consumable' (0)
# Fails if expired, extreme temps (> 32C or < 2C), high humidity (> 75%), or random batch defects
df["is_consumable"] = 1
df.loc[df["days_to_expiry"] <= 0, "is_consumable"] = 0
df.loc[(df["temp_celsius"] > 32) | (df["temp_celsius"] < 2), "is_consumable"] = 0
df.loc[df["humidity_pct"] > 75, "is_consumable"] = 0

# Add 3% random real-world manufacturing/composition defect noise
noise_idx = np.random.choice(
    df.index, size=int(n_samples * 0.03), replace=False
)
df.loc[noise_idx, "is_consumable"] = 0

# 2. Train-Test Split
X = df.drop(columns=["is_consumable"])
y = df["is_consumable"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build Preprocessor & Random Forest Pipeline
categorical_features = ["category", "dosage_form", "manufacturer"]
numeric_features = ["strength_mg", "temp_celsius", "humidity_pct", "days_to_expiry"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("num", StandardScaler(), numeric_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(n_estimators=150, random_state=42),
        ),
    ]
)

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(f"Validation Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Save pipeline and sample data for the app
joblib.dump(pipeline, "medicine_quality_pipeline.pkl")
df.to_csv("medicine_training_data.csv", index=False)
print("Pipeline saved as 'medicine_quality_pipeline.pkl'.")
