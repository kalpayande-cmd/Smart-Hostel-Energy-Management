import sqlite3

DATABASE_NAME = "smart_hostel.db"


def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():

    conn = get_db_connection()
    cursor = conn.cursor()

    # ---------------------------------------------------------
    # SENSOR DATA TABLE
    # ---------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            device_id TEXT NOT NULL,

            temperature REAL,
            humidity REAL,

            pir INTEGER,

            ultrasonic_distance REAL,
            distance_change REAL,
            entry_exit_event TEXT,

            occupancy_count INTEGER,

            mq135_raw REAL,
            mq135_change REAL,
            gas_rise_streak INTEGER,

            light_level REAL,

            current REAL,
            power REAL,
            energy_wh REAL,

            fan_state INTEGER,
            light_state INTEGER,

            ml_occupancy_probability REAL,
            ml_occupancy_prediction INTEGER,
            ml_confidence REAL,
            ml_energy_status TEXT,
            ml_energy_confidence REAL,

            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ---------------------------------------------------------
    # ALERTS TABLE
    # ---------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            device_id TEXT NOT NULL,

            alert_type TEXT NOT NULL,

            message TEXT NOT NULL,

            current REAL,
            power REAL,

            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
        # ---------------------------------------------------------
        # ML COLUMNS
        # ---------------------------------------------------------

    cursor.execute("PRAGMA table_info(sensor_data)")

    existing_columns = [
        row["name"]
        for row in cursor.fetchall()
    ]


    # Occupancy ML confidence
    if "ml_confidence" not in existing_columns:

        cursor.execute("""
            ALTER TABLE sensor_data
            ADD COLUMN ml_confidence REAL
        """)


    # Energy ML status
    if "ml_energy_status" not in existing_columns:

        cursor.execute("""
            ALTER TABLE sensor_data
            ADD COLUMN ml_energy_status TEXT
        """)

    # Energy ML prediction
    if "ml_energy_prediction" not in existing_columns:

        cursor.execute("""
            ALTER TABLE sensor_data
            ADD COLUMN ml_energy_prediction TEXT
        """)


    # Energy ML confidence
    if "ml_energy_confidence" not in existing_columns:

        cursor.execute("""
            ALTER TABLE sensor_data
            ADD COLUMN ml_energy_confidence REAL
        """)

    conn.commit()
    conn.close()

    print("Database initialized successfully.")