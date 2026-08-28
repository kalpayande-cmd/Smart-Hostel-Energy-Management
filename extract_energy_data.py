import sqlite3
import pandas as pd

DB_FILE = "smart_hostel.db"
OUTPUT_FILE = "energy_training_data.csv"

conn = sqlite3.connect(DB_FILE)

query = """
SELECT
    timestamp,
    temperature,
    humidity,
    pir,
    ultrasonic_distance,
    distance_change,
    entry_exit_event,
    occupancy_count,
    mq135_raw,
    mq135_change,
    gas_rise_streak,
    current,
    power,
    energy_wh,
    fan_state,
    light_state
FROM sensor_data
ORDER BY id ASC
"""

df = pd.read_sql_query(query, conn)

conn.close()

print("\n======================================")
print("SMART HOSTEL ENERGY DATA EXTRACTION")
print("======================================")

print("\nRows extracted:")
print(len(df))

print("\nColumns:")
print(list(df.columns))

print("\nFirst 10 records:")
print(df.head(10))

print("\nPower statistics:")
print(df["power"].describe())

print("\nCurrent statistics:")
print(df["current"].describe())

# Save extracted data
df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDataset saved as:")
print(OUTPUT_FILE)