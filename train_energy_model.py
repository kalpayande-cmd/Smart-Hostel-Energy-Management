import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "energy_training_data_ml.csv"
MODEL_FILE = "energy_model.pkl"

FEATURES = [
    "current",
    "power",
    "occupancy_count",
    "fan_state",
    "light_state"
]

TARGET = "energy_status"


print("=" * 55)
print("SMART HOSTEL ENERGY ML TRAINING")
print("=" * 55)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATASET)

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(df[TARGET].value_counts())


# ============================================================
# PREPARE DATA
# ============================================================

X = df[FEATURES]
y = df[TARGET]


print("\nFeatures used by Energy ML:")

for feature in FEATURES:
    print(" -", feature)

print("\nTarget:")
print(" -", TARGET)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Energy ML model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n" + "=" * 55)
print("ENERGY ML PERFORMANCE")
print("=" * 55)

print(
    f"\nAccuracy : {accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=["LOW", "NORMAL", "WASTAGE"],
        zero_division=0
    )
)


print("\nConfusion Matrix:")

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=["LOW", "NORMAL", "WASTAGE"]
)

print(
    pd.DataFrame(
        matrix,
        index=[
            "Actual LOW",
            "Actual NORMAL",
            "Actual WASTAGE"
        ],
        columns=[
            "Predicted LOW",
            "Predicted NORMAL",
            "Predicted WASTAGE"
        ]
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature Importance:")

print(
    importance.to_string(index=False)
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print("\n" + "=" * 55)
print("ENERGY MODEL SAVED")
print("=" * 55)

print("\nSaved as:")
print(MODEL_FILE)

print("\nEnergy ML training completed successfully.")