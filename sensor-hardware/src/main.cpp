#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_NeoPixel.h>
#include <SparkFun_Bio_Sensor_Hub_Library.h>

// ---------------------------------------------------------------------------
// Pins
// ---------------------------------------------------------------------------
const uint8_t PIXEL_PIN       = 11;  // NeoPixel data pin
const uint8_t PIXEL_POWER_PIN = 12;  // NeoPixel power pin

const uint8_t HUB_RESET_PIN = A2;    // -> RST on the pulse oximeter
const uint8_t HUB_MFIO_PIN  = A3;    // -> MFIO on the pulse oximeter

const uint8_t GSR_PIN     = A0;      // -> SIG (yellow) on the Grove GSR sensor
const uint8_t GSR_SAMPLES = 8;       // readings averaged per sample to smooth out noise

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
const unsigned long SAMPLE_INTERVAL_MS = 100;  // 10 Hz processing and output
const unsigned long BEAT_FLASH_MS      = 80;   // how long the pixel lights per beat

const unsigned long HAND_ON_MS  = 1000;   // contact must hold this long to start a session
const unsigned long HAND_OFF_MS = 1500;   // contact must be lost this long to end it
const unsigned long SETTLE_MS   = 5000;   // ignored at session start while electrodes settle
const unsigned long BASELINE_MS = 10000;  // then used to measure the person's baseline

// ---------------------------------------------------------------------------
// Signal processing
// ---------------------------------------------------------------------------
const int   GSR_CONTACT_MARGIN = 30;    // raw drop below the open reading that counts as skin contact
const float GSR_OPEN_TRACKING  = 0.02;  // how fast the open reading follows drift while idle
const float TONIC_TAU_S        = 5.0;   // slow smoothing: overall sweat level
const float FAST_TAU_S         = 0.5;   // fast smoothing: used to detect spikes
const unsigned long TREND_WINDOW_MS = 10000;  // trend = change of the tonic level over this window

// ---------------------------------------------------------------------------
// Interpretation thresholds
// ---------------------------------------------------------------------------
const float LEVEL_PCT         = 10.0;  // GSR change vs. baseline that counts as aroused / calmer
const float HR_AROUSED_BPM    = 10.0;  // heart rate rise vs. baseline that counts as aroused
const float TREND_PCT_PER_MIN = 10.0;  // GSR trend that counts as stressing / relaxing
const float SPIKE_PCT         = 3.0;   // fast GSR jump (% of baseline) that counts as a spike
const unsigned long SPIKE_HOLD_MS = 3000;  // how long "spike" stays set after one is detected

const uint8_t HR_MIN_CONFIDENCE      = 90;  // heart rate readings below this are ignored
const uint8_t STATUS_FINGER_DETECTED = 3;   // finger status reported by the sensor hub

// ---------------------------------------------------------------------------

const uint16_t BASELINE_SAMPLES = BASELINE_MS / SAMPLE_INTERVAL_MS;
const uint16_t TREND_SAMPLES    = TREND_WINDOW_MS / SAMPLE_INTERVAL_MS;

enum Phase { IDLE, BASELINE, MEASURING };
const char *PHASE_NAMES[] = { "idle", "baseline", "measuring" };

Adafruit_NeoPixel pixel(1, PIXEL_PIN, NEO_GRB + NEO_KHZ800);
SparkFun_Bio_Sensor_Hub bioHub(HUB_RESET_PIN, HUB_MFIO_PIN);
bioData body;

Phase phase = IDLE;
unsigned long lastSample = 0;
unsigned long lastBeat   = 0;

// Hand detection
bool handSeen = false;
unsigned long handFirstSeen = 0;
unsigned long handLastSeen  = 0;
float gsrOpen = 0;  // raw GSR reading with nothing touching the electrodes

// Session
unsigned long sessionStart = 0;
float gsrFast    = 0;
float gsrTonic   = 0;
float gsrBase    = NAN;
float hrBase     = NAN;
float lastGoodHr = NAN;

float baselineGsr[BASELINE_SAMPLES];
float baselineHr[BASELINE_SAMPLES];
uint16_t baselineGsrCount = 0;
uint16_t baselineHrCount  = 0;

float tonicHistory[TREND_SAMPLES];
uint16_t historyIndex = 0;
uint16_t historyCount = 0;

uint16_t spikeCount = 0;
unsigned long lastSpike = 0;
bool spikeArmed = true;

// Latest interpretation, NAN / nullptr until measuring
float gsrChange = NAN;
float gsrTrend  = NAN;
float gsrPhasic = NAN;
float hrChange  = NAN;
const char *levelLabel = nullptr;
const char *trendLabel = nullptr;

int readGsr() {
  long sum = 0;
  for (uint8_t i = 0; i < GSR_SAMPLES; i++) {
    sum += analogRead(GSR_PIN);
  }
  return sum / GSR_SAMPLES;
}

// The hub queues a new result for every sample it takes. Read the whole queue
// so `body` always holds the newest result instead of a stale one.
void readLatestBpm() {
  uint8_t pending = bioHub.numSamplesOutFifo();
  while (pending-- > 0) {
    body = bioHub.readBpm();
  }
}

float median(float *values, uint16_t count) {
  if (count == 0) return NAN;
  for (uint16_t i = 1; i < count; i++) {  // insertion sort, fine for ~100 values
    float v = values[i];
    int16_t j = i - 1;
    while (j >= 0 && values[j] > v) {
      values[j + 1] = values[j];
      j--;
    }
    values[j + 1] = v;
  }
  return (count % 2) ? values[count / 2] : (values[count / 2 - 1] + values[count / 2]) / 2;
}

// ---------------------------------------------------------------------------
// JSON output helpers
// ---------------------------------------------------------------------------
void printKey(const char *key) {
  Serial.print(",\"");
  Serial.print(key);
  Serial.print("\":");
}

void printNum(const char *key, float value, uint8_t decimals = 1) {
  printKey(key);
  if (isnan(value)) Serial.print("null");
  else Serial.print(value, decimals);
}

void printInt(const char *key, long value) {
  printKey(key);
  Serial.print(value);
}

void printStr(const char *key, const char *value) {
  printKey(key);
  if (value == nullptr) {
    Serial.print("null");
  } else {
    Serial.print('"');
    Serial.print(value);
    Serial.print('"');
  }
}

void beginEvent(const char *name, unsigned long now) {
  Serial.print("{\"event\":\"");
  Serial.print(name);
  Serial.print('"');
  printInt("t", now);
}

void fail(const char *message) {
  beginEvent("error", millis());
  printStr("msg", message);
  Serial.println('}');
  pixel.setPixelColor(0, pixel.Color(255, 0, 0));  // solid red = error
  pixel.show();
  while (true) {}
}

// ---------------------------------------------------------------------------
// Session handling
// ---------------------------------------------------------------------------
void startBaseline(unsigned long now, float level) {
  phase = BASELINE;
  sessionStart = now;
  gsrFast = gsrTonic = level;
  gsrBase = hrBase = lastGoodHr = NAN;
  baselineGsrCount = baselineHrCount = 0;
  historyIndex = historyCount = 0;
  spikeCount = 0;
  spikeArmed = true;
  gsrChange = gsrTrend = gsrPhasic = hrChange = NAN;
  levelLabel = trendLabel = nullptr;

  beginEvent("session_start", now);
  Serial.println('}');
}

void finishBaseline(unsigned long now) {
  gsrBase = median(baselineGsr, baselineGsrCount);
  hrBase  = median(baselineHr, baselineHrCount);
  phase = MEASURING;

  beginEvent("baseline_done", now);
  printNum("gsr_base", gsrBase);
  printNum("hr_base", hrBase);
  Serial.println('}');
}

void endSession(unsigned long now) {
  beginEvent("session_end", now);
  printNum("duration_s", (now - sessionStart) / 1000.0);
  printNum("gsr_base", gsrBase);
  printNum("gsr_change", gsrChange);
  printNum("hr_base", hrBase);
  printNum("hr_change", hrChange);
  printInt("spikes", spikeCount);
  Serial.println('}');

  phase = IDLE;
}

// ---------------------------------------------------------------------------
// One processing step, every SAMPLE_INTERVAL_MS
// ---------------------------------------------------------------------------
void processSample(unsigned long now) {
  int gsrRaw = readGsr();
  bool fingerOn = body.status == STATUS_FINGER_DETECTED;

  // Learn the "nothing touching" reading while idle. Skin contact lowers it.
  if (phase == IDLE) {
    if (gsrRaw > gsrOpen) gsrOpen = gsrRaw;
    else if (gsrOpen - gsrRaw <= GSR_CONTACT_MARGIN) gsrOpen += (gsrRaw - gsrOpen) * GSR_OPEN_TRACKING;
  }
  bool gsrContact = gsrOpen - gsrRaw > GSR_CONTACT_MARGIN;

  // A hand counts only when both sensors agree, debounced both ways
  bool handNow = fingerOn && gsrContact;
  if (handNow) {
    if (!handSeen) handFirstSeen = now;
    handLastSeen = now;
  }
  handSeen = handNow;

  // Skin conductance proxy: higher = more sweat = more arousal
  float level = gsrOpen - gsrRaw;

  if (phase == IDLE && handSeen && now - handFirstSeen >= HAND_ON_MS) {
    startBaseline(now, level);
  } else if (phase != IDLE && now - handLastSeen >= HAND_OFF_MS) {
    endSession(now);
  }

  if (fingerOn && body.confidence >= HR_MIN_CONFIDENCE && body.heartRate > 0) {
    lastGoodHr = body.heartRate;
  }

  if (phase != IDLE) {
    float dt = SAMPLE_INTERVAL_MS / 1000.0;
    gsrFast  += (level - gsrFast)  * dt / (FAST_TAU_S + dt);
    gsrTonic += (level - gsrTonic) * dt / (TONIC_TAU_S + dt);

    // Tonic value from TREND_WINDOW_MS ago, read before it is overwritten
    bool haveTonicThen = historyCount == TREND_SAMPLES;
    float tonicThen = tonicHistory[historyIndex];
    tonicHistory[historyIndex] = gsrTonic;
    historyIndex = (historyIndex + 1) % TREND_SAMPLES;
    if (historyCount < TREND_SAMPLES) historyCount++;

    unsigned long elapsed = now - sessionStart;

    if (phase == BASELINE && elapsed >= SETTLE_MS) {
      if (baselineGsrCount < BASELINE_SAMPLES) baselineGsr[baselineGsrCount++] = gsrTonic;
      if (baselineHrCount < BASELINE_SAMPLES && !isnan(lastGoodHr)) baselineHr[baselineHrCount++] = lastGoodHr;
      if (elapsed >= SETTLE_MS + BASELINE_MS) finishBaseline(now);
    }

    if (phase == MEASURING && gsrBase > 0) {
      gsrChange = (gsrTonic - gsrBase) / gsrBase * 100;
      gsrPhasic = (gsrFast - gsrTonic) / gsrBase * 100;
      if (haveTonicThen) {
        gsrTrend = (gsrTonic - tonicThen) / gsrBase * 100 * (60000.0 / TREND_WINDOW_MS);
      }
      hrChange = (!isnan(hrBase) && !isnan(lastGoodHr)) ? lastGoodHr - hrBase : NAN;

      // Count a spike on the rising edge, re-arm once it has settled again
      if (spikeArmed && gsrPhasic > SPIKE_PCT) {
        spikeCount++;
        lastSpike = now;
        spikeArmed = false;
      } else if (gsrPhasic < SPIKE_PCT / 2) {
        spikeArmed = true;
      }

      if (gsrChange > LEVEL_PCT || (!isnan(hrChange) && hrChange > HR_AROUSED_BPM)) levelLabel = "aroused";
      else if (gsrChange < -LEVEL_PCT) levelLabel = "calmer";
      else levelLabel = "neutral";

      if (isnan(gsrTrend)) trendLabel = nullptr;
      else if (gsrTrend > TREND_PCT_PER_MIN) trendLabel = "stressing";
      else if (gsrTrend < -TREND_PCT_PER_MIN) trendLabel = "relaxing";
      else trendLabel = "steady";
    }
  }

  // One JSON line per sample
  Serial.print("{\"t\":");
  Serial.print(now);
  printStr("phase", PHASE_NAMES[phase]);
  printInt("hand", handSeen);
  printNum("session_s", phase == IDLE ? NAN : (now - sessionStart) / 1000.0);
  printInt("finger", body.status);
  printInt("hr", body.heartRate);
  printInt("hr_conf", body.confidence);
  printInt("spo2", body.oxygen);
  printInt("gsr_raw", gsrRaw);
  printNum("gsr_open", gsrOpen);
  printNum("gsr", phase == IDLE ? NAN : gsrTonic);
  printNum("gsr_base", gsrBase);
  printNum("gsr_change", gsrChange);
  printNum("gsr_trend", gsrTrend);
  printNum("gsr_phasic", gsrPhasic, 2);
  printNum("hr_base", hrBase);
  printNum("hr_change", hrChange);
  printInt("spikes", spikeCount);
  printInt("spike", phase == MEASURING && spikeCount > 0 && now - lastSpike < SPIKE_HOLD_MS);
  printStr("level", levelLabel);
  printStr("trend", trendLabel);
  Serial.println('}');
}

// Commands from the Mac: 'r' restarts the baseline of the current session
void handleCommands(unsigned long now) {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'r' && phase != IDLE) startBaseline(now, gsrTonic);
  }
}

void updatePixel(unsigned long now) {
  if (body.status != STATUS_FINGER_DETECTED || body.heartRate == 0) {
    pixel.setPixelColor(0, pixel.Color(0, 0, 30));  // dim blue = waiting for a hand
    pixel.show();
    return;
  }

  // Flash once per beat, the colour shows the state
  unsigned long beatInterval = 60000UL / body.heartRate;
  if (now - lastBeat >= beatInterval) lastBeat = now;

  uint32_t color;
  if (phase == IDLE) color = pixel.Color(255, 0, 40);                               // pink: no session yet
  else if (levelLabel == nullptr) color = pixel.Color(255, 255, 255);               // white: baseline
  else if (strcmp(levelLabel, "aroused") == 0) color = pixel.Color(255, 60, 0);     // orange
  else if (strcmp(levelLabel, "calmer") == 0) color = pixel.Color(0, 255, 40);      // green
  else color = pixel.Color(0, 120, 255);                                            // blue: neutral

  if (now - lastBeat < BEAT_FLASH_MS) pixel.setPixelColor(0, color);
  else pixel.clear();
  pixel.show();
}

void setup() {
  pinMode(PIXEL_POWER_PIN, OUTPUT);
  digitalWrite(PIXEL_POWER_PIN, HIGH);  // power on the NeoPixel
  pixel.begin();
  pixel.setBrightness(50);

  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}  // wait briefly for the serial monitor

  Wire.begin();
  if (bioHub.begin() != 0) fail("Sensor hub not found - check wiring");
  if (bioHub.configBpm(MODE_ONE) != 0) fail("Could not configure sensor hub");  // MODE_ONE: HR, SpO2, confidence, status

  gsrOpen = readGsr();  // assumes nobody is wearing the GSR electrodes at power-up

  beginEvent("ready", millis());
  Serial.println('}');
}

void loop() {
  unsigned long now = millis();

  readLatestBpm();
  handleCommands(now);

  if (now - lastSample >= SAMPLE_INTERVAL_MS) {
    lastSample = now;
    processSample(now);
  }

  updatePixel(now);
}
