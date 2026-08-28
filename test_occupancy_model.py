import joblib
import pandas as pd


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_FILE = "occupancy_model.pkl"

model_package = joblib.load(MODEL_FILE)

model = model_package["model"]
features = model_package["features"]

print("\n======================================")
print("SMART HOSTEL ML MODEL TEST")
print("======================================")

print("\nModel features:")
for feature in features:
    print(" -", feature)


# ============================================================
# TEST FUNCTION
# ============================================================

def test_scenario(name, sensor_data):

    # Convert dictionary into DataFrame
    input_data = pd.DataFrame(
        [sensor_data],
        columns=features
    )

    # Prediction
    prediction = model.predict(
        input_data
    )[0]

    # Probability
    probabilities = model.predict_proba(
        input_data
    )[0]

    empty_probability = probabilities[0]
    occupied_probability = probabilities[1]

    print("\n--------------------------------------")
    print(name)
    print("--------------------------------------")

    print("Sensor values:")

    for key, value in sensor_data.items():
        print(
            f"  {key}: {value}"
        )

    print("\nPrediction:")

    if prediction == 1:
        print("  OCCUPIED")
    else:
        print("  EMPTY")

    print(
        f"\nEmpty confidence    : "
        f"{empty_probability * 100:.2f}%"
    )

    print(
        f"Occupied confidence : "
        f"{occupied_probability * 100:.2f}%"
    )


# ============================================================
# SCENARIO 1 — ACTIVE PERSON
# ============================================================

test_scenario(
    "SCENARIO 1 — ACTIVE PERSON",
    {
        "pir": 1,
        "ultrasonic_distance": 150,
        "distance_change": -35,
        "entry_exit_numeric": 0,
        "mq135_raw": 1500,
        "mq135_change": 18,
        "gas_rise_streak": 6
    }
)


# ============================================================
# SCENARIO 2 — SLEEPING / STATIONARY PERSON
# ============================================================

test_scenario(
    "SCENARIO 2 — SLEEPING / STATIONARY PERSON",
    {
        "pir": 0,
        "ultrasonic_distance": 200,
        "distance_change": 1,
        "entry_exit_numeric": 0,
        "mq135_raw": 1450,
        "mq135_change": 9,
        "gas_rise_streak": 10
    }
)


# ============================================================
# SCENARIO 3 — EMPTY ROOM
# ============================================================

test_scenario(
    "SCENARIO 3 — EMPTY ROOM",
    {
        "pir": 0,
        "ultrasonic_distance": 205,
        "distance_change": 0,
        "entry_exit_numeric": 0,
        "mq135_raw": 1200,
        "mq135_change": -5,
        "gas_rise_streak": 0
    }
)


# ============================================================
# SCENARIO 4 — FALSE PIR
# ============================================================

test_scenario(
    "SCENARIO 4 — FALSE PIR",
    {
        "pir": 1,
        "ultrasonic_distance": 200,
        "distance_change": 1,
        "entry_exit_numeric": 0,
        "mq135_raw": 1250,
        "mq135_change": -2,
        "gas_rise_streak": 0
    }
)


# ============================================================
# SCENARIO 5 — PERSON ENTERING
# ============================================================

test_scenario(
    "SCENARIO 5 — PERSON ENTERING",
    {
        "pir": 1,
        "ultrasonic_distance": 110,
        "distance_change": -60,
        "entry_exit_numeric": 1,
        "mq135_raw": 1400,
        "mq135_change": 12,
        "gas_rise_streak": 4
    }
)


# ============================================================
# SCENARIO 6 — PERSON EXITING
# ============================================================

test_scenario(
    "SCENARIO 6 — PERSON EXITING",
    {
        "pir": 0,
        "ultrasonic_distance": 220,
        "distance_change": 60,
        "entry_exit_numeric": -1,
        "mq135_raw": 1400,
        "mq135_change": -2,
        "gas_rise_streak": 1
    }
)


print("\n======================================")
print("MODEL TEST COMPLETED")
print("======================================")