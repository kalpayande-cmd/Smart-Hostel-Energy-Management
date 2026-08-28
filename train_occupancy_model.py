import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# LOAD DATA
# ============================================================

DATASET = "occupancy_training_data_realistic.csv"

df = pd.read_csv(DATASET)

print("\n======================================")
print("SMART HOSTEL OCCUPANCY ML EXPERIMENT")
print("======================================")

print("\nDataset shape:")
print(df.shape)


TARGET = "true_occupancy"


# ============================================================
# MODEL A — CORE SENSOR FUSION
# ============================================================

FEATURES_A = [
    "pir",
    "ultrasonic_distance",
    "distance_change",
    "mq135_raw",
    "mq135_change",
    "gas_rise_streak"
]


# ============================================================
# MODEL B — ENHANCED SENSOR FUSION
# ============================================================

FEATURES_B = [
    "pir",
    "ultrasonic_distance",
    "distance_change",
    "entry_exit_event",
    "mq135_raw",
    "mq135_change",
    "gas_rise_streak"
]


# ============================================================
# CONVERT ENTRY/EXIT TO NUMERIC FEATURES
# ============================================================

# NONE = 0
# ENTER = 1
# EXIT = -1

df["entry_exit_numeric"] = (
    df["entry_exit_event"]
    .map({
        "NONE": 0,
        "ENTER": 1,
        "EXIT": -1
    })
)


# Replace the categorical feature name
FEATURES_B = [
    "pir",
    "ultrasonic_distance",
    "distance_change",
    "entry_exit_numeric",
    "mq135_raw",
    "mq135_change",
    "gas_rise_streak"
]


y = df[TARGET]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_index, test_index = train_test_split(
    range(len(df)),
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_and_evaluate(
    model_name,
    features
):

    print("\n")
    print("======================================")
    print(model_name)
    print("======================================")

    X = df[features]

    X_train = X.iloc[list(train_index)]
    X_test = X.iloc[list(test_index)]

    y_train = y.iloc[list(train_index)]
    y_test = y.iloc[list(test_index)]

    print("\nFeatures:")

    for feature in features:
        print(" -", feature)

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced"
    )

    print("\nTraining...")

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n--------------------------------------")
    print("PERFORMANCE")
    print("--------------------------------------")

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : {f1 * 100:.2f}%"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "EMPTY",
                "OCCUPIED"
            ]
        )
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance = pd.DataFrame({

        "feature": features,

        "importance":
            model.feature_importances_

    })

    importance = importance.sort_values(
        by="importance",
        ascending=False
    )

    print("\nFeature Importance:")

    print(
        importance.to_string(
            index=False
        )
    )

    return model, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


# ============================================================
# MODEL A
# ============================================================

model_a, metrics_a = train_and_evaluate(
    "MODEL A — CORE SENSOR FUSION",
    FEATURES_A
)


# ============================================================
# MODEL B
# ============================================================

model_b, metrics_b = train_and_evaluate(
    "MODEL B — ENHANCED SENSOR FUSION",
    FEATURES_B
)


# ============================================================
# COMPARISON
# ============================================================

print("\n")
print("======================================")
print("MODEL COMPARISON")
print("======================================")

comparison = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],

    "Model A": [
        metrics_a["accuracy"],
        metrics_a["precision"],
        metrics_a["recall"],
        metrics_a["f1"],
        metrics_a["roc_auc"]
    ],

    "Model B": [
        metrics_b["accuracy"],
        metrics_b["precision"],
        metrics_b["recall"],
        metrics_b["f1"],
        metrics_b["roc_auc"]
    ]

})


print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# SAVE ENHANCED MODEL
# ============================================================

model_package = {

    "model": model_b,

    "features": FEATURES_B,

    "feature_description": {

        "pir":
            "PIR motion detection",

        "ultrasonic_distance":
            "Distance measured near doorway",

        "distance_change":
            "Change in ultrasonic distance",

        "entry_exit_numeric":
            "ENTER=1, NONE=0, EXIT=-1",

        "mq135_raw":
            "Raw MQ135 sensor value",

        "mq135_change":
            "Change in MQ135 reading",

        "gas_rise_streak":
            "Consecutive MQ135 rising readings"
    }

}


joblib.dump(
    model_package,
    "occupancy_model.pkl"
)


print("\n======================================")
print("FINAL MODEL SAVED")
print("======================================")

print(
    "occupancy_model.pkl"
)

print("\nExperiment completed successfully.")