/*
  Smart Hostel Occupancy & Energy - ESP32 sensor/actuator node (Flask edition)

  This supersedes the earlier confidence-score version. Your teammate's
  spec makes the ESP32 a dumb sense+report+actuate node: read sensors,
  send the exact JSON schema they specified, and apply whatever
  fan_state/light_state comes back. All occupancy/energy ML now lives in
  Flask, per their note "don't worry about implementing ML logic on ESP32."

  Sensors / actuators (pins unchanged from before):
    HC-SR04 ultrasonic   (TRIG 26, ECHO 25)
    HC-SR501 PIR         (OUT 27)
    MQ135                 (A0 34)
    DHT11                 (DATA 4)
    ACS712                (OUT 35)
    2ch relay, active-low (light IN1 -> GPIO13, fan IN2 -> GPIO14)

  *** ASSUMPTION - confirm with your teammate ***
  Their doc specifies the outgoing JSON exactly but never says HOW Flask
  sends fan_state/light_state commands back. Assumed here: Flask's HTTP
  RESPONSE to each POST contains those two fields, e.g.
  {"fan_state":1,"light_state":0}, applied right after sending. If Flask
  actually expects something else (separate endpoint, polling, MQTT),
  only applyCommandFromResponse() needs to change.

  *** WIRING CHANGE (same as before, still needed) ***
  Move the light relay's IN1 wire from GPIO12 to GPIO13. GPIO12 is an
  ESP32 boot-strapping pin (sets flash voltage); a relay board's pull
  resistor on IN1 can pull it HIGH at boot and hang the board.
*/

#include <DHT.h>
#include <WiFi.h>
#include <HTTPClient.h>

// ---------------- IDENTITY ----------------
// Every ESP32 you deploy needs a UNIQUE id - a hostel means one unit per
// room, and Flask can't tell rooms apart if they all say ESP32_001.
const char* DEVICE_ID = "ESP32_001"; // TODO: change per unit

// ---------------- PINS ----------------
#define TRIG_PIN      26
#define ECHO_PIN      25
#define PIR_PIN       27
#define MQ135_PIN     34
#define DHT_PIN       4
#define ACS712_PIN    35
#define RELAY_LIGHT   13   // moved off GPIO12 - rewire required
#define RELAY_FAN     14
#define DHTTYPE DHT11
DHT dht(DHT_PIN, DHTTYPE);

const int RELAY_ON  = HIGH;  // HIGH-trigger relay
const int RELAY_OFF = LOW;   // LOW releases relay

// ---------------- WIFI / FLASK ----------------
const char* WIFI_SSID     = "POCO M5";       // TODO
const char* WIFI_PASSWORD = "0987654321@";   // TODO
const char* FLASK_URL = "http://10.125.198.117:5000/api/sensor-data";
// const char* FLASK_URL ="http://192.168.0.105:5000/api/sensor-data"; // TODO: your laptop's LAN IP
const unsigned long REPORT_INTERVAL_MS = 2000; // tune with your teammate - faster = more responsive ML, more network/DHT load

// ---------------- ULTRASONIC ----------------
const int DOOR_BASELINE_CM = 200; // CALIBRATE against your actual mounting
const int ENTRY_DROP_CM    = 50;  // CALIBRATE
float previousDistance = DOOR_BASELINE_CM;

// ---------------- PIR (edge-latched between reports) ----------------
bool motionSinceLastReport = false;

// ---------------- MQ135 ----------------
int previousMQ135 = 0;
int mq135RiseStreak = 0;

// ---------------- CURRENT ----------------
// ASSUMES the ACS712-5A variant (185mV/A, 2.5V zero). Change if yours is
// the 20A (100mV/A) or 30A (66mV/A) board.
const float ACS712_MV_PER_AMP = 185.0;
const float ACS712_ZERO_V = 2.5;
const float SUPPLY_VOLTAGE = 5.0; // DC prototype; real AC needs true RMS + power factor, not just V*I

// ---------------- RELAY STATE ----------------
int fanState = 0;
int lightState = 0;

// ---------------- OCCUPANCY (kept simple, per your teammate's note) ----------------
int occupancyCount = 0;

unsigned long lastReportMs = 0;

void setup() {
  Serial.begin(115200);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(PIR_PIN, INPUT);
  pinMode(RELAY_LIGHT, OUTPUT);
  pinMode(RELAY_FAN, OUTPUT);
  applyRelayState(1, 1); // start both ON

  dht.begin();

  Serial.print("Warming up MQ135 (60s)");
  unsigned long warmStart = millis();
  while (millis() - warmStart < 60000) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println(" done.");
  previousMQ135 = analogRead(MQ135_PIN);

  connectWiFi();
  lastReportMs = millis();
  Serial.println("System ready.");
}

void loop() {
  // Poll PIR every iteration so a brief pulse between report cycles isn't
  // missed - latched until the next report, then cleared. Note: this loop
  // still blocks during the HTTP POST inside runReportCycle(), so motion
  // during that window won't be caught until the loop resumes. Fine at a
  // 2s+ cadence; would need async HTTP or a second core to fix properly.
  if (digitalRead(PIR_PIN) == HIGH) {
    motionSinceLastReport = true;
  }

  unsigned long now = millis();
  if (now - lastReportMs >= REPORT_INTERVAL_MS) {
    lastReportMs = now;
    runReportCycle();
  }
}

void runReportCycle() {
  // ---- sample sensors ----
  float distance = pingDistanceCm();
  if (distance < 0) distance = previousDistance; // timeout - reuse last good value instead of reporting garbage
  float distanceChange = distance - previousDistance;

  int pir = motionSinceLastReport ? 1 : 0;
  motionSinceLastReport = false;

  int mq135Raw = analogRead(MQ135_PIN);
  int mq135Change = mq135Raw - previousMQ135;
  mq135RiseStreak = (mq135Change > 0) ? (mq135RiseStreak + 1) : 0;

  float t = dht.readTemperature();
  float h = dht.readHumidity();
  if (isnan(t)) t = -1; // flag a failed read instead of sending "nan" and breaking Flask's JSON parser
  if (isnan(h)) h = -1;

  float amps = readCurrentAmps();
  float watts = amps * SUPPLY_VOLTAGE;

  String entryExit = determineEntryExit(distanceChange, distance);
  if (entryExit == "ENTER") occupancyCount++;
  else if (entryExit == "EXIT") occupancyCount = max(0, occupancyCount - 1);

  // ---- build + send JSON ----
  String payload = buildPayload(t, h, pir, distance, distanceChange, entryExit,
                                 occupancyCount, mq135Raw, mq135Change, mq135RiseStreak,
                                 amps, watts);
  String response;
  bool ok = postToFlask(payload, response);
  if (ok) applyCommandFromResponse(response);

  // ---- roll state forward for next cycle's deltas ----
  previousDistance = distance;
  previousMQ135 = mq135Raw;
}

// ================= DERIVED FIELDS =================
String determineEntryExit(float distanceChange, float currentDistance) {
  // Best-effort only, per your teammate's note - Flask/ML refines this.
  // If entry/exit come out swapped on your mounting, flip the two branches.
  if (distanceChange < -ENTRY_DROP_CM && currentDistance < DOOR_BASELINE_CM - ENTRY_DROP_CM) {
    return "ENTER";
  } else if (distanceChange > ENTRY_DROP_CM && currentDistance > DOOR_BASELINE_CM - 15) {
    return "EXIT";
  }
  return "NONE";
}

// ================= SENSORS =================
long pingDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH, 20000);
  if (duration == 0) return -1;
  return duration * 0.034 / 2;
}

float readCurrentAmps() {
  int adc = analogRead(ACS712_PIN);
  // Undo the 1k/2k divider (ratio 2/3) before applying the ACS712 spec -
  // skipping this understates current by ~33%.
  float voltage = ((adc / 4095.0) * 3.3) * 1.5;
  float ampsVal = (voltage - ACS712_ZERO_V) / (ACS712_MV_PER_AMP / 1000.0);
  return abs(ampsVal);
}

// ================= RELAY =================
void applyRelayState(int light, int fan) {
  lightState = light;
  fanState = fan;
  digitalWrite(RELAY_LIGHT, light ? RELAY_ON : RELAY_OFF);
  digitalWrite(RELAY_FAN, fan ? RELAY_ON : RELAY_OFF);
  Serial.print("Relay GPIO states -> LIGHT: ");
  Serial.print(digitalRead(RELAY_LIGHT));
  Serial.print(" FAN: ");
  Serial.println(digitalRead(RELAY_FAN));
}

void applyCommandFromResponse(const String& response) {
  int newLight = extractIntField(response, "light_state");
  int newFan   = extractIntField(response, "fan_state");
  // Only act on fields that actually parsed - if Flask's reply is missing
  // one or malformed, leave that relay exactly as it was. Guessing here
  // is worse than doing nothing.
  if (newLight < 0) newLight = lightState;
  if (newFan   < 0) newFan   = fanState;
  if (newLight != lightState || newFan != fanState) {
    applyRelayState(newLight, newFan);
    Serial.print("Applied command -> light="); Serial.print(newLight);
    Serial.print(" fan="); Serial.println(newFan);
  }
}

// Tiny hand-rolled extractor so we don't need ArduinoJson just to read two
// small integer fields out of a known, fixed response shape.
int extractIntField(const String& json, const String& key) {
  String pattern = "\"" + key + "\":";
  int idx = json.indexOf(pattern);
  if (idx < 0) return -1;
  idx += pattern.length();
  while (idx < (int)json.length() && json[idx] == ' ') idx++;
  int end = idx;
  while (end < (int)json.length() && (isDigit(json[end]) || json[end] == '-')) end++;
  if (end == idx) return -1;
  return json.substring(idx, end).toInt();
}

// ================= JSON PAYLOAD =================
String buildPayload(float t, float h, int pir, float distance, float distanceChange,
                     const String& entryExit, int occCount, int mq135Raw, int mq135Change,
                     int riseStreak, float amps, float watts) {
  String p = "{";
  p += "\"device_id\":\"" + String(DEVICE_ID) + "\",";
  p += "\"temperature\":" + String(t, 1) + ",";
  p += "\"humidity\":" + String(h, 1) + ",";
  p += "\"pir\":" + String(pir) + ",";
  p += "\"ultrasonic_distance\":" + String(distance, 2) + ",";
  p += "\"distance_change\":" + String(distanceChange, 2) + ",";
  p += "\"entry_exit_event\":\"" + entryExit + "\",";
  p += "\"occupancy_count\":" + String(occCount) + ",";
  p += "\"mq135_raw\":" + String(mq135Raw) + ",";
  p += "\"mq135_change\":" + String(mq135Change) + ",";
  p += "\"gas_rise_streak\":" + String(riseStreak) + ",";
  p += "\"current\":" + String(amps, 2) + ",";
  p += "\"power\":" + String(watts, 2) + ",";
  p += "\"fan_state\":" + String(fanState) + ",";
  p += "\"light_state\":" + String(lightState);
  p += "}";
  return p;
}

// ================= WIFI / HTTP =================
void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
    delay(500);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print(" connected, IP=");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println(" failed - will retry next report cycle.");
  }
}

bool postToFlask(const String& payload, String& response) {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
    if (WiFi.status() != WL_CONNECTED) return false;
  }
  HTTPClient http;
  http.begin(FLASK_URL);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(payload);
  bool ok = (code == 200);
  if (ok) {
    response = http.getString();
  } else {
    Serial.printf("POST to Flask failed, HTTP %d\n", code);
  }
  http.end();
  return ok;
}
