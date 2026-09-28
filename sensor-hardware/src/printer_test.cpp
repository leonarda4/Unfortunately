// Thermal printer test: prints a header, then a strip chart of simulated heart
// rate and GSR data at the same 10 Hz rate as the real firmware, then a footer
// with timing results. Needs no sensors.
//
//   pio run -e printer_test -t upload
//   pio device monitor              # status once per second, send p to print again

#include <Arduino.h>
#include <ThermalStrip.h>

const uint32_t PRINTER_BAUD = 19200;  // printed on the self-test page (hold feed while powering on)

const unsigned long SAMPLE_INTERVAL_MS = 100;    // same as the real firmware
const unsigned long TEST_DURATION_MS   = 30000;
const float HR_RANGE_BPM = 30;   // lane shows baseline +- this
const float GSR_RANGE_PCT = 30;  // lane shows baseline +- this

// Simulated GSR spikes, in seconds after the chart starts
const float SPIKE_TIMES[] = { 6, 13, 21 };
const uint8_t SPIKE_MARK_SAMPLES = 5;  // marker length per spike

enum State { HEADER, CHART, FOOTER, DONE };

ThermalPrinter printer;
StripChart chart(printer, 2);  // 2 dot rows per sample: 0.25 mm, i.e. 2.5 mm of paper per second

State state = HEADER;
unsigned long lastSample = 0;
unsigned long chartStart = 0;
unsigned long lastReport = 0;

// Timing results
unsigned long maxLateMs = 0;    // how late a 10 Hz tick ran
unsigned long maxUpdateUs = 0;  // longest printer.update() call
uint16_t maxQueued = 0;
uint16_t dropped = 0;

float gsrChange = 0;

void printHeader() {
  printer.align('C');
  printer.style(true, true);
  printer.text("PRINTER TEST\n");
  printer.style(false, false);
  printer.text("simulated data, 10 Hz\n");
  printer.feed(1);

  // 32 characters per line; each lane is half the paper width
  printer.align('L');
  printer.style(true, false);
  printer.text("HEART RATE      SKIN (GSR)\n");
  printer.style(false, false);
  printer.text("-30     0   +30  -30    0    +30\n");
  printer.text("bpm vs baseline  % vs baseline\n");
}

void printFooter() {
  char line[40];
  printer.feed(1);
  printer.style(true, false);
  printer.text("Test finished\n");
  printer.style(false, false);
  snprintf(line, sizeof line, "max tick late:  %lu ms\n", maxLateMs);
  printer.text(line);
  snprintf(line, sizeof line, "max update:     %lu us\n", maxUpdateUs);
  printer.text(line);
  snprintf(line, sizeof line, "dropped samples: %u\n", dropped);
  printer.text(line);
  printer.feed(4);
}

void startTest() {
  maxLateMs = maxUpdateUs = 0;
  maxQueued = dropped = 0;
  gsrChange = 0;
  printHeader();
  state = HEADER;
}

float noise(float amount) {
  return random(-1000, 1001) / 1000.0 * amount;
}

// Heart rate wanders around the baseline and rises at the end. GSR drifts up
// with a few spikes and finally runs past the lane to test clipping.
void simulate(float t, float *hr, float *gsr, bool *spike) {
  *hr = 6 * sin(TWO_PI * t / 12) + noise(1.5);
  if (t > 20) *hr += (t - 20) * 2.5;

  float dt = SAMPLE_INTERVAL_MS / 1000.0;
  gsrChange += 0.6 * dt;                   // slow drift
  if (t > 24) gsrChange += 6 * dt;         // runs out of range at the end
  gsrChange -= gsrChange * 0.02 * dt;      // slight pull back to baseline
  *gsr = gsrChange + noise(0.4);

  *spike = false;
  for (float s : SPIKE_TIMES) {
    if (t >= s) *gsr += 9 * exp(-(t - s) / 1.5);  // quick rise, slow decay
    if (t >= s && t < s + SPIKE_MARK_SAMPLES * dt) *spike = true;
  }
}

void sampleTick(unsigned long now) {
  if (state == HEADER && printer.idle()) {
    chart.start();
    chartStart = now;
    state = CHART;
  }

  if (state == CHART) {
    float t = (now - chartStart) / 1000.0;
    float values[2];
    bool spike;
    simulate(t, &values[0], &values[1], &spike);
    if (!chart.add(values, spike ? 0b10 : 0)) dropped++;

    if (now - chartStart >= TEST_DURATION_MS) {
      printFooter();
      state = FOOTER;
    }
  } else if (state == FOOTER && printer.idle()) {
    Serial.println("done - send p to print again");
    state = DONE;
  }
}

void report() {
  Serial.print("state=");
  Serial.print(state == HEADER ? "header" : state == CHART ? "chart" : state == FOOTER ? "footer" : "done");
  Serial.print(" queued=");
  Serial.print(printer.queued());
  Serial.print("B max_queued=");
  Serial.print(maxQueued);
  Serial.print("B max_late=");
  Serial.print(maxLateMs);
  Serial.print("ms max_update=");
  Serial.print(maxUpdateUs);
  Serial.print("us dropped=");
  Serial.println(dropped);
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}

  printer.begin(Serial1, PRINTER_BAUD);

  // Lanes line up with the header text: 12 dots per character
  chart.addLane(12, 180, -HR_RANGE_BPM, HR_RANGE_BPM);
  chart.addLane(204, 372, -GSR_RANGE_PCT, GSR_RANGE_PCT);
  chart.setGrid(10, 100);  // tick every second, line every 10 s

  randomSeed(analogRead(A1));
  startTest();
  Serial.println("printer test started");
}

void loop() {
  unsigned long now = millis();

  unsigned long before = micros();
  printer.update();
  maxUpdateUs = max(maxUpdateUs, micros() - before);
  maxQueued = max(maxQueued, printer.queued());

  if (Serial.read() == 'p' && state == DONE) startTest();

  if (now - lastSample >= SAMPLE_INTERVAL_MS) {
    if (lastSample != 0 && state == CHART) maxLateMs = max(maxLateMs, now - lastSample - SAMPLE_INTERVAL_MS);
    lastSample = now;
    sampleTick(now);
  }

  if (now - lastReport >= 1000) {
    lastReport = now;
    if (state != DONE) report();
  }
}
