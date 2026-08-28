import joblib
import pandas as pd


MODEL_FILE = "energy_model.pkl"


print("=" * 55)
print("SMART HOSTEL ENERGY ML MODEL TEST")
print("=" * 55)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_FILE)

print("\nEnergy ML model loaded successfully.")

print("\nModel features:")
for feature in model.feature_names_in_:
    print(" -", feature)


# ============================================================
# TEST FUNCTION
# ============================================================

def test_scenario(name, current, power, occupancy, fan, light):

    data = pd.DataFrame([{
        "current": current,
        "power": power,
        "occupancy_count": occupancy,
        "fan_state": fan,
        "light_state": light
    }])

    prediction = model.predict(data)[0]

    probabilities = model.predict_proba(data)[0]

    classes = model.classes_

    probability_dict = dict(
        zip(classes, probabilities)
    )

    confidence = max(probabilities)

    print("\n" + "-" * 55)
    print(name)
    print("-" * 55)

    print("Sensor / system state:")
    print("  Current          :", current, "A")
    print("  Power            :", power, "W")
    print("  Occupancy count  :", occupancy)
    print("  Fan state        :", fan)
    print("  Light state      :", light)

    print("\nPrediction:")
    print(" ", prediction)

    print("\nProbabilities:")

    for class_name in classes:
        print(
            f"  {class_name:<10}: "
            f"{probability_dict[class_name] * 100:.2f}%"
        )

    print(
        f"\nConfidence: {confidence * 100:.2f}%"
    )


# ============================================================
# SCENARIO 1
# EMPTY ROOM + APPLIANCES OFF
# ============================================================

test_scenario(
    "SCENARIO 1 — EMPTY ROOM + APPLIANCES OFF",
    current=0.02,
    power=0.24,
    occupancy=0,
    fan=0,
    light=0
)


# ============================================================
# SCENARIO 2
# EMPTY ROOM + FAN ON
# ============================================================

test_scenario(
    "SCENARIO 2 — EMPTY ROOM + FAN ON",
    current=0.60,
    power=7.20,
    occupancy=0,
    fan=1,
    light=0
)


# ============================================================
# SCENARIO 3
# EMPTY ROOM + FAN + LIGHT ON
# ============================================================

test_scenario(
    "SCENARIO 3 — EMPTY ROOM + FAN + LIGHT ON",
    current=0.80,
    power=9.60,
    occupancy=0,
    fan=1,
    light=1
)


# ============================================================
# SCENARIO 4
# OCCUPIED + FAN ON
# ============================================================

test_scenario(
    "SCENARIO 4 — OCCUPIED + FAN ON",
    current=0.60,
    power=7.20,
    occupancy=1,
    fan=1,
    light=0
)


# ============================================================
# SCENARIO 5
# OCCUPIED + FAN + LIGHT ON
# ============================================================

test_scenario(
    "SCENARIO 5 — OCCUPIED + FAN + LIGHT ON",
    current=0.80,
    power=9.60,
    occupancy=1,
    fan=1,
    light=1
)


# ============================================================
# SCENARIO 6
# OCCUPIED + APPLIANCES OFF
# ============================================================

test_scenario(
    "SCENARIO 6 — OCCUPIED + APPLIANCES OFF",
    current=0.02,
    power=0.24,
    occupancy=1,
    fan=0,
    light=0
)


print("\n" + "=" * 55)
print("ENERGY ML TEST COMPLETED")
print("=" * 55)