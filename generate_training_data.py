import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 30000

data = []

for _ in range(N_SAMPLES):

    # ========================================================
    # TRUE OCCUPANCY
    # ========================================================

    true_occupancy = np.random.choice(
        [0, 1],
        p=[0.50, 0.50]
    )

    # ========================================================
    # ENVIRONMENT
    # ========================================================

    temperature = np.random.normal(29, 4)
    humidity = np.random.normal(62, 12)

    temperature = np.clip(temperature, 18, 40)
    humidity = np.clip(humidity, 25, 95)

    # ========================================================
    # DEFAULT SENSOR VALUES
    # ========================================================

    pir = 0

    ultrasonic_distance = np.random.normal(200, 30)

    distance_change = np.random.normal(0, 8)

    entry_exit_event = "NONE"

    mq135_raw = np.random.normal(1350, 180)

    mq135_change = np.random.normal(0, 15)

    gas_rise_streak = 0

    # ========================================================
    # EMPTY ROOM
    # ========================================================

    if true_occupancy == 0:

        mq135_raw = np.random.normal(1300, 220)

        mq135_change = np.random.normal(-2, 18)

        gas_rise_streak = np.random.choice(
            [0, 1, 2, 3, 4, 5],
            p=[0.40, 0.20, 0.15, 0.10, 0.10, 0.05]
        )

        # False PIR
        pir = np.random.choice(
            [0, 1],
            p=[0.82, 0.18]
        )

        # Door movement without occupancy change
        door_event = np.random.choice(
            ["NONE", "ENTER", "EXIT"],
            p=[0.88, 0.06, 0.06]
        )

        if door_event == "ENTER":

            entry_exit_event = "ENTER"

            distance_change = np.random.normal(
                -45,
                25
            )

        elif door_event == "EXIT":

            entry_exit_event = "EXIT"

            distance_change = np.random.normal(
                45,
                25
            )

        ultrasonic_distance = np.random.normal(
            200,
            45
        )

    # ========================================================
    # OCCUPIED ROOM
    # ========================================================

    else:

        mq135_raw = np.random.normal(
            1450,
            260
        )

        mq135_change = np.random.normal(
            10,
            20
        )

        scenario = np.random.choice(
            [
                "active",
                "stationary",
                "sleeping",
                "entry",
                "exit",
                "weak_signal"
            ],
            p=[
                0.30,
                0.25,
                0.15,
                0.10,
                0.05,
                0.15
            ]
        )

        # ====================================================
        # ACTIVE
        # ====================================================

        if scenario == "active":

            pir = np.random.choice(
                [0, 1],
                p=[0.15, 0.85]
            )

            distance_change = np.random.normal(
                0,
                30
            )

            mq135_change = np.random.normal(
                15,
                20
            )

            gas_rise_streak = np.random.randint(
                1,
                10
            )

        # ====================================================
        # STATIONARY
        # ====================================================

        elif scenario == "stationary":

            pir = np.random.choice(
                [0, 1],
                p=[0.85, 0.15]
            )

            distance_change = np.random.normal(
                0,
                8
            )

            mq135_change = np.random.normal(
                15,
                15
            )

            gas_rise_streak = np.random.randint(
                3,
                12
            )

        # ====================================================
        # SLEEPING
        # ====================================================

        elif scenario == "sleeping":

            pir = np.random.choice(
                [0, 1],
                p=[0.97, 0.03]
            )

            distance_change = np.random.normal(
                0,
                4
            )

            mq135_change = np.random.normal(
                8,
                12
            )

            gas_rise_streak = np.random.randint(
                4,
                15
            )

        # ====================================================
        # ENTRY
        # ====================================================

        elif scenario == "entry":

            pir = np.random.choice(
                [0, 1],
                p=[0.35, 0.65]
            )

            entry_exit_event = "ENTER"

            distance_change = np.random.normal(
                -55,
                25
            )

            ultrasonic_distance = np.random.normal(
                110,
                35
            )

            mq135_change = np.random.normal(
                10,
                15
            )

            gas_rise_streak = np.random.randint(
                0,
                6
            )

        # ====================================================
        # EXIT
        # ====================================================

        elif scenario == "exit":

            pir = np.random.choice(
                [0, 1],
                p=[0.45, 0.55]
            )

            entry_exit_event = "EXIT"

            distance_change = np.random.normal(
                55,
                25
            )

            ultrasonic_distance = np.random.normal(
                220,
                40
            )

            mq135_change = np.random.normal(
                2,
                18
            )

            gas_rise_streak = np.random.randint(
                0,
                6
            )

        # ====================================================
        # WEAK SIGNAL
        # ====================================================

        elif scenario == "weak_signal":

            pir = np.random.choice(
                [0, 1],
                p=[0.60, 0.40]
            )

            distance_change = np.random.normal(
                0,
                15
            )

            mq135_change = np.random.normal(
                5,
                22
            )

            gas_rise_streak = np.random.randint(
                0,
                7
            )

    # ========================================================
    # REALISTIC SENSOR NOISE
    # ========================================================

    ultrasonic_distance += np.random.normal(
        0,
        15
    )

    distance_change += np.random.normal(
        0,
        5
    )

    mq135_raw += np.random.normal(
        0,
        50
    )

    mq135_change += np.random.normal(
        0,
        5
    )

    # ========================================================
    # SENSOR FAILURES
    # ========================================================

    if np.random.random() < 0.04:
        pir = 0

    if np.random.random() < 0.03:

        ultrasonic_distance += np.random.normal(
            0,
            80
        )

    if np.random.random() < 0.03:

        mq135_raw += np.random.normal(
            0,
            180
        )

    # ========================================================
    # LIMIT VALUES
    # ========================================================

    ultrasonic_distance = np.clip(
        ultrasonic_distance,
        20,
        400
    )

    distance_change = np.clip(
        distance_change,
        -150,
        150
    )

    mq135_raw = np.clip(
        mq135_raw,
        500,
        3000
    )

    mq135_change = np.clip(
        mq135_change,
        -100,
        100
    )

    # ========================================================
    # STORE SAMPLE
    # ========================================================

    data.append({

        "temperature": round(
            temperature,
            2
        ),

        "humidity": round(
            humidity,
            2
        ),

        "pir": pir,

        "ultrasonic_distance": round(
            ultrasonic_distance,
            2
        ),

        "distance_change": round(
            distance_change,
            2
        ),

        "entry_exit_event": entry_exit_event,

        "mq135_raw": round(
            mq135_raw,
            2
        ),

        "mq135_change": round(
            mq135_change,
            2
        ),

        "gas_rise_streak": gas_rise_streak,

        "true_occupancy": true_occupancy
    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(data)


# ============================================================
# SAVE REALISTIC DATASET
# ============================================================

OUTPUT_FILE = "occupancy_training_data_realistic.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# OUTPUT
# ============================================================

print("\n======================================")
print("REALISTIC SMART HOSTEL ML DATASET")
print("======================================")

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(
    df["true_occupancy"].value_counts()
)

print("\nColumns:")
print(
    list(df.columns)
)

print("\nFirst 10 samples:")
print(
    df.head(10)
)

print("\nSaved as:")
print(OUTPUT_FILE)