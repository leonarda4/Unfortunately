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
|---|---|
| 3V3 | 3V |
| GND | GND |
| SDA | SDA (A4) |
| SCL | SCL (A5) |
| RST | A2 |
| MFIO | A3 |

A Qwiic/STEMMA QT cable covers 3V3, GND, SDA and SCL. RST and MFIO need two
extra jumper wires. Use 3.3 V only.

| Grove GSR wire | QT Py M0 |
|---|---|
| Yellow (SIG) | A0 |
| White (NC) | not connected |
| Red (VCC) | 3V |
| Black (GND) | GND |

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

### How a session works

| Phase | What happens | LED |
|---|---|---|
| idle | Waits until both sensors detect skin for 1 s | dim blue |
| baseline | 5 s settling, then 10 s to measure the person's resting level | white flash per beat |
| measuring | Everything reported relative to the baseline | beat flash: orange aroused, blue neutral, green calmer |
| end | Hand removed for 1.5 s: summary sent, back to idle | |

Thresholds and durations are constants at the top of `src/main.cpp`. They are
starting values and should be tuned with recorded sessions.

### Serial output

115200 baud, one JSON object per line at 10 Hz. Main fields:

| Field | Meaning |
|---|---|
| `phase` | `idle`, `baseline` or `measuring` |
| `hr`, `hr_conf`, `spo2` | heart rate (bpm), its confidence (%), SpO2 (%) |
| `gsr_raw` | raw GSR reading (lower = more sweat) |
| `gsr` | smoothed GSR level (higher = more sweat) |
| `gsr_change` | GSR level vs. baseline, in % |
| `gsr_trend` | GSR change rate over the last 10 s, in %/min |
| `hr_change` | heart rate vs. baseline, in bpm |
| `spike`, `spikes` | sudden GSR jump right now / count this session |
| `level` | `aroused`, `neutral` or `calmer` |
| `trend` | `stressing`, `steady` or `relaxing` |

Events are separate lines with an `event` key: `ready`, `session_start`,
`baseline_done`, `session_end` (with a session summary) and `error`.
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
