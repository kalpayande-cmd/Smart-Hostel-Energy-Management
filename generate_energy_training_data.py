import pandas as pd
import numpy as np

print("=" * 50)
print("SMART HOSTEL ENERGY TRAINING DATA GENERATOR")
print("=" * 50)


# ============================================================
# LOAD REAL ENERGY DATA
# ============================================================

REAL_DATASET = "energy_training_data.csv"
OUTPUT_DATASET = "energy_training_data_ml.csv"

real_df = pd.read_csv(REAL_DATASET)

print("\nReal dataset shape:")
print(real_df.shape)


# ============================================================
# KEEP USEFUL REAL DATA
# ============================================================

real_columns = [
    "current",
    "power",
    "occupancy_count",
    "fan_state",
    "light_state"
]

real_df = real_df[real_columns].copy()


# ============================================================
# LABEL REAL DATA
# ============================================================

def classify_real_data(row):

    occupancy = int(row["occupancy_count"])
    power = float(row["power"])

    # Empty room consuming significant power
    if occupancy == 0 and power >= 2.0:
        return "WASTAGE"

    # Occupied room with almost no power
    if occupancy > 0 and power < 0.5:
        return "LOW"

    return "NORMAL"


real_df["energy_status"] = real_df.apply(
    classify_real_data,
    axis=1
)


# ============================================================
# GENERATE REALISTIC SYNTHETIC SCENARIOS
# ============================================================

np.random.seed(42)


def generate_scenario(
    count,
    occupancy,
    fan,
    light,
    power_min,
    power_max,
    status
):

    rows = []

    for _ in range(count):

        power = np.random.uniform(
            power_min,
            power_max
        )

        # Your current sensor data follows approximately:
        #
        # Power = Voltage × Current
        #
        # Your dummy data uses ~12 V.
        #
        # Therefore:
        #
        # Current = Power / 12

        current = power / 12.0

        # Small measurement noise
        current += np.random.normal(
            0,
            0.005
        )

        current = max(
            0,
            current
        )

        rows.append({
            "current": round(current, 3),
            "power": round(power, 3),
            "occupancy_count": occupancy,
            "fan_state": fan,
            "light_state": light,
            "energy_status": status
        })

    return pd.DataFrame(rows)


# ============================================================
# SCENARIO 1
# EMPTY + APPLIANCES OFF
#
# Expected:
# NORMAL
# ============================================================

empty_normal = generate_scenario(
    count=1500,
    occupancy=0,
    fan=0,
    light=0,
    power_min=0.0,
    power_max=0.5,
    status="NORMAL"
)


# ============================================================
# SCENARIO 2
# OCCUPIED + FAN ON
#
# Expected:
# NORMAL
# ============================================================

occupied_fan = generate_scenario(
    count=1500,
    occupancy=1,
    fan=1,
    light=0,
    power_min=5.0,
    power_max=8.5,
    status="NORMAL"
)


# ============================================================
# SCENARIO 3
# OCCUPIED + FAN + LIGHT ON
#
# Expected:
# NORMAL
# ============================================================

occupied_both = generate_scenario(
    count=1500,
    occupancy=1,
    fan=1,
    light=1,
    power_min=7.0,
    power_max=12.0,
    status="NORMAL"
)


# ============================================================
# SCENARIO 4
# EMPTY + FAN ON
#
# Expected:
# WASTAGE
# ============================================================

empty_fan = generate_scenario(
    count=1500,
    occupancy=0,
    fan=1,
    light=0,
    power_min=4.5,
    power_max=9.0,
    status="WASTAGE"
)


# ============================================================
# SCENARIO 5
# EMPTY + LIGHT ON
#
# Expected:
# WASTAGE
# ============================================================

empty_light = generate_scenario(
    count=1500,
    occupancy=0,
    fan=0,
    light=1,
    power_min=3.0,
    power_max=7.0,
    status="WASTAGE"
)


# ============================================================
# SCENARIO 6
# EMPTY + FAN + LIGHT ON
#
# Expected:
# WASTAGE
# ============================================================

empty_both = generate_scenario(
    count=1500,
    occupancy=0,
    fan=1,
    light=1,
    power_min=7.0,
    power_max=13.0,
    status="WASTAGE"
)


# ============================================================
# SCENARIO 7
# OCCUPIED + APPLIANCES OFF
#
# Expected:
# LOW
#
# This is not necessarily "wastage".
# It simply means the room is occupied but
# very little electrical load is being consumed.
# ============================================================

occupied_low = generate_scenario(
    count=1500,
    occupancy=1,
    fan=0,
    light=0,
    power_min=0.0,
    power_max=0.4,
    status="LOW"
)


# ============================================================
# COMBINE EVERYTHING
# ============================================================

synthetic_df = pd.concat(
    [
        empty_normal,
        occupied_fan,
        occupied_both,
        empty_fan,
        empty_light,
        empty_both,
        occupied_low
    ],
    ignore_index=True
)


# ============================================================
# COMBINE REAL + SYNTHETIC
# ============================================================

final_df = pd.concat(
    [
        real_df,
        synthetic_df
    ],
    ignore_index=True
)


# ============================================================
# SHUFFLE
# ============================================================

final_df = final_df.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    OUTPUT_DATASET,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 50)
print("ENERGY ML DATASET GENERATED")
print("=" * 50)

print("\nDataset shape:")
print(final_df.shape)

print("\nClass distribution:")
print(
    final_df["energy_status"].value_counts()
)

print("\nSample records:")

print(
    final_df.head(10).to_string(
        index=False
    )
)

print("\nDataset saved as:")
print(OUTPUT_DATASET)

print("\nEnergy dataset generation completed.")