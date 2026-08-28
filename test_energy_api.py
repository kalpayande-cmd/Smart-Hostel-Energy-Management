import requests


URL = "http://127.0.0.1:5000/api/sensor-data"


def send_test(name, data):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("\nSending:")
    print(data)

    try:

        response = requests.post(
            URL,
            json=data,
            timeout=5
        )

        print("\nHTTP Status:", response.status_code)

        print("\nServer Response:")
        print(response.json())

    except requests.exceptions.ConnectionError:

        print("\nERROR: Flask server is not running.")

    except Exception as e:

        print("\nERROR:", e)


# ============================================================
# TEST 1 — EMPTY ROOM + APPLIANCES OFF
# Expected Energy ML: NORMAL
# ============================================================

send_test(
    "TEST 1 — EMPTY + APPLIANCES OFF",

    {
        "device_id": "ENERGY_TEST_001",

        "temperature": 28.0,
        "humidity": 60.0,

        "pir": 0,

        "ultrasonic_distance": 220.0,
        "distance_change": 0.0,

        "entry_exit_event": "NONE",

        "occupancy_count": 0,

        "mq135_raw": 1200,
        "mq135_change": 0,
        "gas_rise_streak": 0,

        "current": 0.02,

        "fan_state": 0,
        "light_state": 0
    }
)


# ============================================================
# TEST 2 — EMPTY ROOM + FAN ON
# Expected Energy ML: WASTAGE
# ============================================================

send_test(
    "TEST 2 — EMPTY + FAN ON",

    {
        "device_id": "ENERGY_TEST_001",

        "temperature": 28.0,
        "humidity": 60.0,

        "pir": 0,

        "ultrasonic_distance": 220.0,
        "distance_change": 0.0,

        "entry_exit_event": "NONE",

        "occupancy_count": 0,

        "mq135_raw": 1200,
        "mq135_change": 0,
        "gas_rise_streak": 0,

        "current": 0.60,

        "fan_state": 1,
        "light_state": 0
    }
)


# ============================================================
# TEST 3 — EMPTY ROOM + FAN + LIGHT ON
# Expected Energy ML: WASTAGE
# ============================================================

send_test(
    "TEST 3 — EMPTY + FAN + LIGHT ON",

    {
        "device_id": "ENERGY_TEST_001",

        "temperature": 28.0,
        "humidity": 60.0,

        "pir": 0,

        "ultrasonic_distance": 220.0,
        "distance_change": 0.0,

        "entry_exit_event": "NONE",

        "occupancy_count": 0,

        "mq135_raw": 1200,
        "mq135_change": 0,
        "gas_rise_streak": 0,

        "current": 0.80,

        "fan_state": 1,
        "light_state": 1
    }
)


# ============================================================
# TEST 4 — OCCUPIED + FAN ON
# Expected Energy ML: NORMAL
# ============================================================

send_test(
    "TEST 4 — OCCUPIED + FAN ON",

    {
        "device_id": "ENERGY_TEST_001",

        "temperature": 30.0,
        "humidity": 65.0,

        "pir": 1,

        "ultrasonic_distance": 120.0,
        "distance_change": -20.0,

        "entry_exit_event": "ENTER",

        "occupancy_count": 1,

        "mq135_raw": 1500,
        "mq135_change": 20,
        "gas_rise_streak": 4,

        "current": 0.60,

        "fan_state": 1,
        "light_state": 0
    }
)


# ============================================================
# TEST 5 — OCCUPIED + APPLIANCES OFF
# Expected Energy ML: LOW
# ============================================================

send_test(
    "TEST 5 — OCCUPIED + APPLIANCES OFF",

    {
        "device_id": "ENERGY_TEST_001",

        "temperature": 30.0,
        "humidity": 65.0,

        "pir": 0,

        "ultrasonic_distance": 150.0,
        "distance_change": 1.0,

        "entry_exit_event": "NONE",

        "occupancy_count": 1,

        "mq135_raw": 1500,
        "mq135_change": 5,
        "gas_rise_streak": 2,

        "current": 0.02,

        "fan_state": 0,
        "light_state": 0
    }
)


print("\n" + "=" * 60)
print("ENERGY API TEST COMPLETED")
print("=" * 60)