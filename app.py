#http://127.0.0.1:5000/api/latest-sensor-data
#http://127.0.0.1:5000/api/energy-summary
#http://127.0.0.1:5000/api/wastage-status?utm_source=chatgpt.com
#http://127.0.0.1:5000/
import joblib
import os
import pandas as pd
from flask import Flask, request, jsonify,render_template
from database import initialize_database, get_db_connection

app = Flask(__name__)        #creates a new instance of the Flask web application.
# ============================================================
# LOAD OCCUPANCY ML MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "occupancy_model.pkl"
)

occupancy_model = None
occupancy_features = None

try:
    model_package = joblib.load(MODEL_PATH)

    occupancy_model = model_package["model"]
    occupancy_features = model_package["features"]

    print("======================================")
    print("OCCUPANCY ML MODEL LOADED")
    print("======================================")
    print("Features:")

    for feature in occupancy_features:
        print(" -", feature)

except Exception as e:

    print("======================================")
    print("WARNING: OCCUPANCY ML MODEL NOT LOADED")
    print("======================================")
    print("Error:", e)
    # ============================================================
# LOAD ENERGY ML MODEL
# ============================================================

ENERGY_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "energy_model.pkl"
)

energy_model = None
energy_features = None

try:

    energy_model = joblib.load(ENERGY_MODEL_PATH)

    energy_features = [
        "current",
        "power",
        "occupancy_count",
        "fan_state",
        "light_state"
    ]

    print("======================================")
    print("ENERGY ML MODEL LOADED")
    print("======================================")

    print("Features:")

    for feature in energy_features:
        print(" -", feature)

except Exception as e:

    print("======================================")
    print("WARNING: ENERGY ML MODEL NOT LOADED")
    print("======================================")

    print("Error:", e)
    # ============================================================
    # OCCUPANCY ML PREDICTION
    # ============================================================

def predict_occupancy_ml(data):

    if occupancy_model is None:
        return {
            "prediction": None,
            "occupancy_probability": None,
            "empty_probability": None,
            "confidence": None,
            "status": "MODEL_NOT_LOADED"
        }

    try:

        # ----------------------------------------------------
        # Convert entry/exit event to numeric value
        # ----------------------------------------------------

        event = data.get(
            "entry_exit_event",
            "NONE"
        )

        event_mapping = {
            "NONE": 0,
            "ENTER": 1,
            "EXIT": -1
        }

        entry_exit_numeric = event_mapping.get(
            event,
            0
        )

        # ----------------------------------------------------
        # Prepare ML input
        # ----------------------------------------------------

        ml_input = pd.DataFrame([{

            "pir":
                data.get("pir", 0),

            "ultrasonic_distance":
                data.get(
                    "ultrasonic_distance",
                    0
                ),

            "distance_change":
                data.get(
                    "distance_change",
                    0
                ),

            "entry_exit_numeric":
                entry_exit_numeric,

            "mq135_raw":
                data.get(
                    "mq135_raw",
                    0
                ),

            "mq135_change":
                data.get(
                    "mq135_change",
                    0
                ),

            "gas_rise_streak":
                data.get(
                    "gas_rise_streak",
                    0
                )

        }])

        # Make sure columns are exactly what
        # the model expects

        ml_input = ml_input[
            occupancy_features
        ]

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = occupancy_model.predict(
            ml_input
        )[0]

        probabilities = occupancy_model.predict_proba(
            ml_input
        )[0]

        empty_probability = float(
            probabilities[0]
        )

        occupied_probability = float(
            probabilities[1]
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = max(
            empty_probability,
            occupied_probability
        )

        if occupied_probability >= 0.80:

            status = "HIGH"

        elif occupied_probability <= 0.20:

            status = "HIGH"

        else:

            status = "UNCERTAIN"

        return {

            "prediction":
                "OCCUPIED"
                if prediction == 1
                else "EMPTY",

            "occupancy_probability":
                occupied_probability,

            "empty_probability":
                empty_probability,

            "confidence":
                confidence,

            "status":
                status
        }

    except Exception as e:

        print(
            "ML prediction error:",
            e
        )

        return {

            "prediction": None,

            "occupancy_probability":
                None,

            "empty_probability":
                None,

            "confidence":
                None,

            "status":
                "PREDICTION_ERROR"
        }
    # ============================================================
    # ENERGY ML PREDICTION
    # ============================================================

def predict_energy_ml(data):

    if energy_model is None:

        return {
            "prediction": None,
            "confidence": None,
            "normal_probability": None,
            "wastage_probability": None,
            "low_probability": None,
            "status": "MODEL_NOT_LOADED"
        }

    try:

        ml_input = pd.DataFrame([{

            "current": float(
                data.get("current", 0)
            ),

            "power": float(
            data.get(
            "power",
            float(data.get("current", 0)) * 12.0
                )
            ),

            "occupancy_count": int(
                data.get("occupancy_count", 0)
            ),

            "fan_state": int(
                data.get("fan_state", 0)
            ),

            "light_state": int(
                data.get("light_state", 0)
            )

        }])

        # Make sure columns are exactly what
        # the energy model expects

        ml_input = ml_input[
            energy_features
        ]

        prediction = energy_model.predict(
            ml_input
        )[0]

        probabilities = energy_model.predict_proba(
            ml_input
        )[0]

        probability_map = dict(
            zip(
                energy_model.classes_,
                probabilities
            )
        )

        confidence = float(
            max(probabilities)
        )

        return {

            "prediction":
                str(prediction),

            "confidence":
                confidence,

            "normal_probability":
                float(
                    probability_map.get(
                        "NORMAL",
                        0
                    )
                ),

            "wastage_probability":
                float(
                    probability_map.get(
                        "WASTAGE",
                        0
                    )
                ),

            "low_probability":
                float(
                    probability_map.get(
                        "LOW",
                        0
                    )
                ),

            "status":
                "OK"
        }

    except Exception as e:

        print(
            "Energy ML prediction error:",
            e
        )

        return {

            "prediction": None,

            "confidence": None,

            "normal_probability": None,

            "wastage_probability": None,

            "low_probability": None,

            "status":
                "PREDICTION_ERROR"
        }
empty_room_readings = 0

CURRENT_THRESHOLD = 0.1
EMPTY_ROOM_THRESHOLD = 6

# Initialize database
initialize_database()

# POST: Receive and store sensor data
@app.route("/api/sensor-data", methods=["POST"])   #A POST request is an HTTP method used to send data to a server to create or update a resource.
def receive_sensor_data():

    data = request.get_json()      #Javascript object notation

    if not data:
        return jsonify({
            "success": False,
            "message": "No JSON data received"
        }), 400

    # Extract sensor values
    device_id = data.get("device_id")

    temperature = data.get("temperature")
    humidity = data.get("humidity")

    pir = data.get("pir")

    ultrasonic_distance = data.get(
    "ultrasonic_distance"
    )

    distance_change = data.get(
        "distance_change"
    )

    entry_exit_event = data.get(
        "entry_exit_event"
    )

    occupancy_count = data.get(
        "occupancy_count"
    )

    mq135_raw = data.get(
        "mq135_raw"
    )

    mq135_change = data.get(
        "mq135_change"
    )

    gas_rise_streak = data.get(
        "gas_rise_streak"
    )

    

    current = data.get("current")

    fan_state = data.get(
        "fan_state",
        0
    )

    light_state = data.get(
        "light_state",
        0
    )
    # ============================================================
    # AI OCCUPANCY PREDICTION
    # ============================================================

    ml_result = predict_occupancy_ml(data)

    print("\n======================================")
    print("AI OCCUPANCY PREDICTION")
    print("======================================")

    print(
        "Prediction:",
        ml_result["prediction"]
    )

    print(
        "Occupied probability:",
        f"{ml_result['occupancy_probability'] * 100:.2f}%"
        if ml_result["occupancy_probability"] is not None
        else "N/A"
    )

    print(
        "Confidence:",
        f"{ml_result['confidence'] * 100:.2f}%"
        if ml_result["confidence"] is not None
        else "N/A"
    )

    print(
        "Status:",
        ml_result["status"]
    )
    # ============================================================
    # CALCULATE POWER
    # ============================================================

    # Prototype voltage
    VOLTAGE = 12

    # Calculate power
    power = round(VOLTAGE * float(current or 0), 2)


    # ============================================================
    # AI ENERGY PREDICTION
    # ============================================================

    # Add calculated power to data before sending it to ML
    energy_data = data.copy()
    energy_data["power"] = power

    energy_result = predict_energy_ml(energy_data)

    print("\n======================================")
    print("AI ENERGY PREDICTION")
    print("======================================")

    print(
        "Prediction:",
        energy_result["prediction"]
    )

    print(
        "Confidence:",
        f"{energy_result['confidence'] * 100:.2f}%"
        if energy_result["confidence"] is not None
        else "N/A"
    )

    print(
        "Status:",
        energy_result["status"]
    )
    # Time interval between sensor readings in seconds
    TIME_INTERVAL_SECONDS = 5

    # Convert seconds to hours
    time_hours = TIME_INTERVAL_SECONDS / 3600

    # Energy consumed during this interval in Wh
    energy_wh = round(power * time_hours, 6)
    # ----------------------------------------
    # ENERGY WASTAGE DETECTION
    # ----------------------------------------

    global empty_room_readings

    # Room is empty but electrical power is still being used
    if occupancy_count == 0 and current > CURRENT_THRESHOLD:
        empty_room_readings += 1
    else:
        empty_room_readings = 0

    # Check if empty room condition has continued long enough
    wastage_detected = empty_room_readings >= EMPTY_ROOM_THRESHOLD
        # ============================================================
    # AI APPLIANCE CONTROL DECISION
    # ============================================================

    commanded_fan_state = fan_state
    commanded_light_state = light_state

# ============================================================
# AI APPLIANCE CONTROL
# ============================================================

# CASE 1: Room is EMPTY and energy is being wasted
    if (
        energy_result["prediction"] == "WASTAGE"
        and ml_result["prediction"] == "EMPTY"
    ):
        commanded_fan_state = 0
        commanded_light_state = 0

        print("AI CONTROL: EMPTY + WASTAGE")
        print("AI CONTROL: Turning FAN and LIGHT OFF")


    # CASE 2: Room is OCCUPIED and energy use is NORMAL
    elif (
        ml_result["prediction"] == "OCCUPIED"
        and energy_result["prediction"] == "NORMAL"
    ):
        commanded_fan_state = 1
        commanded_light_state = 1

    print("AI CONTROL: OCCUPIED + NORMAL")
    print("AI CONTROL: Turning FAN and LIGHT ON")
    # Store alert when wastage is first detected
    if empty_room_readings == EMPTY_ROOM_THRESHOLD:

        alert_message = (
            "Potential energy wastage: Room is empty "
            "but electrical power is still being consumed."
        )

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO alerts
            (device_id, alert_type, message, current, power)
            VALUES (?, ?, ?, ?, ?)
        """, (
            device_id,
            "ENERGY_WASTAGE",
            alert_message,
            current,
            power
        ))

        conn.commit()
        conn.close()

        print("⚠ ENERGY WASTAGE ALERT STORED")

    # ============================================================
    # STORE DATA IN SQLITE
    # ============================================================

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO sensor_data
        (
            device_id,
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
            light_state,
            ml_occupancy_probability,
            ml_occupancy_prediction,
            ml_confidence,
            ml_energy_prediction,
            ml_energy_status,
            ml_energy_confidence
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, (
        device_id,
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
        light_state,

        # ML occupancy probability
        ml_result["occupancy_probability"],

        # ML prediction
        1 if ml_result["prediction"] == "OCCUPIED" else 0,

        # ML confidence
        ml_result["confidence"],

        # ML energy prediction
        energy_result["prediction"],

        # ML energy status
        energy_result["status"],

        # ML energy confidence
        energy_result["confidence"]

        ))

    conn.commit()

    record_id = cursor.lastrowid

    conn.close()

    print("\nReceived and Stored Sensor Data:")
    print(data)

    return jsonify({
    "success": True,
    "message": "Sensor data received and stored successfully",

    "record_id": record_id,

    "power_watts": power,
    "energy_wh": energy_wh,

    "wastage_detected": wastage_detected,

    "fan_state": commanded_fan_state,
    "light_state": commanded_light_state,

    # Energy ML
    "energy_ml_prediction":
        energy_result["prediction"],

    "energy_ml_confidence":
        energy_result["confidence"],

    "energy_ml_status":
        energy_result["status"]
    }) 

# GET: Retrieve all stored sensor data
@app.route("/api/sensor-data", methods=["GET"])
def get_sensor_data():

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM sensor_data
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    sensor_records = []

    for row in rows:
        sensor_records.append({
            "id": row["id"],
            "device_id": row["device_id"],

            "temperature": row["temperature"],
            "humidity": row["humidity"],

            "pir": row["pir"],

            "ultrasonic_distance":
                row["ultrasonic_distance"],

            "distance_change":
                row["distance_change"],

            "entry_exit_event":
                row["entry_exit_event"],

            "occupancy_count":
                row["occupancy_count"],

            "mq135_raw":
                row["mq135_raw"],

            "mq135_change":
                row["mq135_change"],

            "gas_rise_streak":
                row["gas_rise_streak"],

            "current": row["current"],
            "power": row["power"],
            "energy_wh": row["energy_wh"],

            "fan_state":
                row["fan_state"],

            "light_state":
                row["light_state"],

            "ml_occupancy_probability":
                row["ml_occupancy_probability"],

            "ml_occupancy_prediction":
                row["ml_occupancy_prediction"],

            "timestamp":
                row["timestamp"]
        })

    return jsonify({
        "success": True,
        "count": len(sensor_records),
        "data": sensor_records
    }), 200
@app.route("/api/latest-sensor-data", methods=["GET"])
def get_latest_sensor_data():

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM sensor_data
        ORDER BY id DESC
        LIMIT 1
    """)

    row = cursor.fetchone()
    conn.close()

    # If there is no sensor data yet
    if row is None:
        return jsonify({
            "success": False,
            "message": "No sensor data found"
        }), 404

    latest_data = {
    "id": row["id"],
    "device_id": row["device_id"],

    "temperature": row["temperature"],
    "humidity": row["humidity"],

    "pir": row["pir"],

    "ultrasonic_distance":
        row["ultrasonic_distance"],

    "distance_change":
        row["distance_change"],

    "entry_exit_event":
        row["entry_exit_event"],

    "occupancy_count":
        row["occupancy_count"],

    "mq135_raw":
        row["mq135_raw"],

    "mq135_change":
        row["mq135_change"],

    "gas_rise_streak":
        row["gas_rise_streak"],

    "current": row["current"],
    "power": row["power"],
    "energy_wh": row["energy_wh"],

    "fan_state":
        row["fan_state"],

    "light_state":
        row["light_state"],

    "ml_occupancy_probability":
        row["ml_occupancy_probability"],

    "ml_occupancy_prediction":
        row["ml_occupancy_prediction"],
        
    "ml_confidence":
        row["ml_confidence"],

    "ml_energy_prediction":
        row["ml_energy_prediction"],


    # Energy ML
    "ml_energy_status":
        row["ml_energy_status"],

    "ml_energy_confidence":
        row["ml_energy_confidence"],       

    "timestamp":
        row["timestamp"]
}
    return jsonify({
        "success": True,
        "data": latest_data
    }), 200
@app.route("/api/energy-summary", methods=["GET"])
def get_energy_summary():

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) AS total_readings,
            COALESCE(SUM(energy_wh), 0) AS total_energy_wh
        FROM sensor_data
    """)

    result = cursor.fetchone()
    conn.close()

    total_readings = result["total_readings"]
    total_energy_wh = round(result["total_energy_wh"], 6)
    total_energy_kwh = round(total_energy_wh / 1000, 6)

    return jsonify({
        "success": True,
        "total_readings": total_readings,
        "total_energy_wh": total_energy_wh,
        "total_energy_kwh": total_energy_kwh
    }), 200
@app.route("/api/wastage-status", methods=["GET"])
def get_wastage_status():

    if empty_room_readings >= EMPTY_ROOM_THRESHOLD:
        message = "Potential energy wastage detected"
        wastage_detected = True
    else:
        message = "No energy wastage currently detected"
        wastage_detected = False

    return jsonify({
        "success": True,
        "empty_room_readings": empty_room_readings,
        "threshold_readings": EMPTY_ROOM_THRESHOLD,
        "wastage_detected": wastage_detected,
        "message": message
    }), 200
@app.route("/api/alerts", methods=["GET"])
def get_alerts():

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM alerts
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    alerts = []

    for row in rows:
        alerts.append({
            "id": row["id"],
            "device_id": row["device_id"],
            "alert_type": row["alert_type"],
            "message": row["message"],
            "current": row["current"],
            "power": row["power"],
            "timestamp": row["timestamp"]
        })

    return jsonify({
        "success": True,
        "count": len(alerts),
        "data": alerts
    }), 200
@app.route("/api/daily-energy", methods=["GET"])
def get_daily_energy():

    conn = get_db_connection()
    cursor = conn.cursor()

    # Get today's energy consumption and reading count
    cursor.execute("""
        SELECT
            COUNT(*) AS total_readings,
            COALESCE(SUM(energy_wh), 0) AS total_energy_wh
        FROM sensor_data
        WHERE DATE(timestamp) = DATE('now')
    """)

    energy_result = cursor.fetchone()

    # Get today's wastage alerts
    cursor.execute("""
        SELECT COUNT(*) AS wastage_alerts
        FROM alerts
        WHERE DATE(timestamp) = DATE('now')
    """)

    alert_result = cursor.fetchone()

    conn.close()

    total_readings = energy_result["total_readings"]
    total_energy_wh = round(energy_result["total_energy_wh"], 6)
    total_energy_kwh = round(total_energy_wh / 1000, 6)

    wastage_alerts = alert_result["wastage_alerts"]

    return jsonify({
        "success": True,
        "total_readings": total_readings,
        "total_energy_wh": total_energy_wh,
        "total_energy_kwh": total_energy_kwh,
        "wastage_alerts": wastage_alerts
    }), 200
@app.route("/")
def dashboard():
    return render_template("dashboard.html")
@app.route("/api/recent-readings", methods=["GET"])
def get_recent_readings():

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT timestamp,
        power,
        current,
        energy_wh,
        temperature,
        humidity,
        mq135_raw,
        pir,
        ultrasonic_distance,
        occupancy_count
        FROM sensor_data
        ORDER BY id DESC
        LIMIT 20
    """)

    rows = cursor.fetchall()
    conn.close()

    readings = []

    # Reverse so charts display oldest → newest
    for row in reversed(rows):
        readings.append({
            "timestamp": row["timestamp"],
            "power": row["power"],
            "current": row["current"],
            "energy_wh": row["energy_wh"],
            "temperature": row["temperature"],
            "humidity": row["humidity"],
            "mq135_raw": row["mq135_raw"],
            "pir": row["pir"],
            "ultrasonic_distance": row["ultrasonic_distance"],
            "occupancy_count": row["occupancy_count"],
        })

    return jsonify({
        "success": True,
        "data": readings
    }), 200
@app.route("/api/dashboard-data", methods=["GET"])
def get_dashboard_data():

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM sensor_data
        ORDER BY id DESC
        LIMIT 1
    """)

    row = cursor.fetchone()

    conn.close()

    if row is None:

        return jsonify({
            "success": False,
            "message": "No sensor data available"
        }), 404

    return jsonify({

        "success": True,

        "data": {

            "temperature":
                row["temperature"],

            "humidity":
                row["humidity"],

            "pir":
                row["pir"],

            "ultrasonic_distance":
                row["ultrasonic_distance"],

            "distance_change":
                row["distance_change"],

            "entry_exit_event":
                row["entry_exit_event"],

            "occupancy_count":
                row["occupancy_count"],

            "mq135_raw":
                row["mq135_raw"],

            "mq135_change":
                row["mq135_change"],

            "gas_rise_streak":
                row["gas_rise_streak"],
            "current":
                row["current"],

            "power":
                row["power"],

            "energy_wh":
                row["energy_wh"],

            "fan_state":
                row["fan_state"],

            "light_state":
                row["light_state"],

            "ml_occupancy_probability":
                row["ml_occupancy_probability"],

            "ml_occupancy_prediction":
                row["ml_occupancy_prediction"],

            "timestamp":
                row["timestamp"]
        }

    }), 200
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True
)