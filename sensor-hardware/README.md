# Sensor hardware

Heart rate, blood oxygen (SpO2) and skin conductance (GSR) measured on an
Adafruit QT Py M0 and streamed to a Mac over USB serial. The firmware detects
when a hand is placed on both sensors, measures a personal baseline, and reports
how arousal changes relative to it. A Python dashboard plots the signals and
shows the interpretation (aroused / neutral / calmer, stressing / steady / relaxing).

## Hardware

- Adafruit QT Py M0 (SAMD21)
- SparkFun Pulse Oximeter and Heart Rate Sensor (MAX30101 + MAX32664, SEN-15219)
- Seeed Grove GSR sensor

### Wiring

| Pulse oximeter | QT Py M0 |
Install the dashboard dependencies in the active Python environment:

```bash
cd sensor-hardware
uv pip install -r requirements.txt
```

Then run the live dashboard from this directory:

|---|---|
python scripts/dashboard.py              # live graphs + interpretation
python scripts/dashboard.py --demo       # simulated data, no board needed
python scripts/dashboard.py --log s.csv  # also save every sample to CSV
python scripts/read_sensor.py            # plain terminal output
| RST | A2 |

To select a specific serial device, pass its port explicitly:

```bash
python scripts/dashboard.py --port /dev/cu.usbmodemXXXX
```

On macOS, list likely QT Py ports with:

```bash
ls /dev/cu.usb*
```
| MFIO | A3 |

A Qwiic/STEMMA QT cable covers 3V3, GND, SDA and SCL. RST and MFIO need two
extra jumper wires. Use 3.3 V only.

| Grove GSR wire | QT Py M0 |
|---|---|
| Yellow (SIG) | A0 |
| White (NC) | not connected |
| Red (VCC) | 3V |
| Black (GND) | GND |

| Thermal printer (Adafruit CSN-A2, TTL) | Connect to |
|---|---|
| Data RX (usually yellow, data into the printer) | QT Py TX (A6) |
| Data TX (usually green) | not connected: it outputs 5 V, which the QT Py can't take |
| Data GND (black) | QT Py GND |
| Power + / GND (red / black) | separate 5–9 V supply, at least 2 A |

Never power the printer from the QT Py or USB: it draws over 1.5 A while
printing. The printer ties both of its GND wires together inside, so the data
GND wire gives the QT Py and the printer supply a common ground. Hold the feed
button while powering the printer on to print a self-test page with its baud
rate (usually 19200, set as `PRINTER_BAUD` in the firmware).

Put the GSR electrodes on two fingers of one hand and the pulse sensor under a
fingertip of the other hand. Power the board up with the GSR straps off: it
learns the "nothing touching" GSR reading at start-up.

## Firmware

A [PlatformIO](https://platformio.org/) project. Open this folder in VS Code with
the PlatformIO extension, or from the command line:

```bash
cd sensor-hardware
pio run -t upload
```

If the upload can't find the board, double-press the QT Py's reset button (the LED
turns green) and upload again.

### Printer test

`src/printer_test.cpp` prints a 30 s strip chart of simulated heart rate and GSR
data without any sensors, then a footer with timing results. The printer code in
`lib/ThermalStrip` queues everything and sends it from `loop()` without blocking.

```bash
pio run -e printer_test -t upload
pio device monitor    # status once per second, send p to print again
```

Upload the main firmware again afterwards with `pio run -t upload`.

### How a session works

| Phase | What happens | LED |
|---|---|---|
| idle | Waits until both sensors detect skin for 1 s | dim blue |
| baseline | 5 s settling, then 10 s to measure the person's resting level | white flash per beat |
| measuring | Everything reported relative to the baseline | beat flash: orange aroused, blue neutral, green calmer |
| hand lifted | Up to 10 s off the sensors counts as the same candidate. Brief losses are ignored; after a longer break the baseline starts over (if still measuring it) or the smoothing restarts | dim blue |
| end | Hand removed for 10 s: summary sent, back to idle. The next hand is a new candidate | |

Each session gets a candidate number (#0001, #0002, ...). It is kept in flash
across power cycles, but uploading firmware starts it at 1 again.

The thermal printer prints, per session:

1. At session start: the candidate number and a legend explaining the graph. This
   takes about as long as the baseline, so the graph follows right below it.
2. While the hand is on the sensors: a strip chart of heart rate change (dotted,
   bpm) and GSR change (solid, %) against the baseline, sharing the full paper
   width and one scale. A change of `PRINT_SCALE` (10) reaches 3/4 of the way to
   the edge; bigger changes are squeezed in near the edges instead of cut off.
   Faint guide lines and the numbers above the graph mark ±5 and ±15. A marker
   at the right edge shows every GSR spike, a dashed line where the hand was
   lifted.
3. At session end: blank paper to tear off.

Thresholds and durations are constants at the top of `src/main.cpp`. They are
starting values and should be tuned with recorded sessions.

### Serial output

115200 baud, one JSON object per line at 10 Hz. Main fields:

| Field | Meaning |
|---|---|
| `phase` | `idle`, `baseline` or `measuring` |
| `candidate` | number of the current (or last) session |
| `away_s` | seconds since the hand was lifted during a session, else `null` |
| `hr`, `hr_conf`, `spo2` | heart rate (bpm), its confidence (%), SpO2 (%) |
| `gsr_raw` | raw GSR reading (lower = more sweat) |
| `gsr` | smoothed GSR level (higher = more sweat) |
| `gsr_change` | GSR level vs. baseline, in % |
| `gsr_trend` | GSR change rate over the last 10 s, in %/min |
| `hr_change` | heart rate vs. baseline, in bpm |
| `spike`, `spikes` | sudden GSR jump right now / count this session |
| `level` | `aroused`, `neutral` or `calmer` |
| `trend` | `stressing`, `steady` or `relaxing` |

Events are separate lines with an `event` key: `ready`, `session_start` (with
the candidate number), `baseline_start`, `baseline_done`, `hand_lifted`,
`hand_returned`, `session_end` (with a session summary) and `error`.
Sending `r` restarts the baseline of the current session.

The GSR value is a relative measure, not calibrated microsiemens. GSR reflects
arousal in general, so excitement, focus and stress look alike.

## Mac side

```bash
pip install pyserial matplotlib
python3 scripts/dashboard.py              # live graphs + interpretation
python3 scripts/dashboard.py --demo       # simulated data, no board needed
python3 scripts/dashboard.py --log s.csv  # also save every sample
python3 scripts/read_sensor.py            # plain terminal output
```

Only one program can use the serial port at a time, so close the PlatformIO serial
monitor first.
