# Sensor hardware

Heart rate, blood oxygen (SpO2) and skin conductance (GSR) measured on an
Adafruit QT Py M0 and streamed to a computer over USB serial. The firmware detects
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

A [PlatformIO](https://platformio.org/) project. It works the same on macOS,
Windows and Linux.

### Setting up a computer (once)

1. Install [VS Code](https://code.visualstudio.com/) and its **PlatformIO IDE**
   extension. Without VS Code, `pip install platformio` installs the `pio`
   command on its own.
2. Open **the `sensor-hardware` folder itself** in VS Code (File → Open Folder…),
   not the whole repository. PlatformIO only recognises the project when
   `platformio.ini` is at the top of the open folder.
3. Build once: `pio run`, or the ✓ button in the PlatformIO bar at the bottom.
   The first build downloads the compiler and libraries (a few hundred MB), so it
   needs internet and takes a few minutes. Building never touches the board.

If a terminal says `pio: command not found`, use the copy the VS Code extension
installed: `~/.platformio/penv/bin/pio` on macOS and Linux,
`%USERPROFILE%\.platformio\penv\Scripts\pio.exe` on Windows. Or use the buttons
in the PlatformIO bar.

- **Windows** needs no driver. The board shows up as a `COM` port.
- **Linux** needs the [PlatformIO udev rules](https://docs.platformio.org/en/latest/core/installation/udev-rules.html)
  and your user in the `dialout` group, otherwise uploads fail with
  "permission denied".
- Use a USB cable that carries data. Many charging cables don't, and the board
  then never shows up.

### Uploading

Close everything that uses the board's serial port first: the dashboard,
`read_sensor.py`, the PlatformIO serial monitor, the Arduino IDE. Only one
program can use the port, and an upload fails while another one holds it.

```bash
cd sensor-hardware
pio run -t upload
```

Or use the → button in the PlatformIO bar. The upload does three things:

1. It asks the running firmware to restart into the **bootloader**, the small
   built-in program that receives new firmware. The LED turns green.
2. It waits for the bootloader to appear as a new USB port, then writes the
   firmware.
3. It checks what it wrote. A good upload ends with `Verify successful` and
   `[SUCCESS]`, then the board restarts by itself and the LED turns dim blue.

The board is found automatically by its USB ID on every operating system. Don't
add a fixed `upload_port` to `platformio.ini`: port names differ per computer
(`/dev/cu.usbmodem1101`, `COM5`, `/dev/ttyACM0`), and a wrong one makes every
upload fail. To choose a board when several are connected, pass the port for that
one upload: `pio run -t upload --upload-port COM5`.

Uploading resets the candidate counter to #0001.

### What the LED means

| QT Py LED | Meaning |
|---|---|
| Dim blue | Firmware running, waiting for a hand |
| Flashing with the heartbeat | Session running (colour: see below) |
| Solid red | Firmware running, but the pulse sensor board wasn't found at start-up. Check its wiring, then press reset once |
| Green | Bootloader: waiting for an upload. **No firmware is running**, so the sensors do nothing |
| Off | No power, or the board is stuck. Unplug it and plug it back in |

### Stuck in the bootloader (green LED)

A failed upload can't break the bootloader, so the board can always be recovered.

1. **Press reset once** (a single press). If a complete firmware is on the board,
   it starts and the LED turns dim blue.
2. **Still green? Upload again** with `pio run -t upload`. If an earlier upload
   broke off halfway, the board stays in the bootloader until one finishes. The
   upload finds the waiting bootloader directly.
3. **Board not showing up at all** (the upload says `Couldn't find a board`):
   unplug it and plug it back in, then **double-press reset quickly**. The LED
   turns green and a drive called `QTPY_BOOT` appears. Run the upload while it's
   green. If nothing changes, try another cable or USB port and avoid USB hubs.

Other common errors:

| Error | Cause |
|---|---|
| `Couldn't find a board on the selected port` | The board didn't come back after the restart, or a fixed `upload_port` doesn't match this computer. Double-press reset and upload again |
| `Resource busy`, `Access is denied`, `could not open port` | Another program has the port open. Close the dashboard or serial monitor |
| `Permission denied` (Linux) | udev rules and `dialout` group missing (see above) |
| Upload stops partway (e.g. at 10 %) | The connection dropped. Upload again. The bootloader keeps waiting |

### Notes for coding agents

For coding agents (Codex, Claude Code and others) working on this folder, and the
people running them:

- Use `pio run` to check that the code compiles. It is safe to run any time.
- **Uploading needs direct USB access, and the first build needs internet.**
  Agent sandboxes often block one or both. Uploads then fail after the restart
  step and leave the board in the bootloader (green LED). Run the upload in a
  normal terminal, or let the agent run it outside its sandbox. Then follow
  "Stuck in the bootloader" above.
- Don't add a fixed `upload_port` or `monitor_port` to `platformio.ini`.
- An agent can't press the reset button. When the board needs a reset or a
  double-press, ask the person at the computer.
- `pio device monitor`, `read_sensor.py` and `dashboard.py` run until stopped.
  Run them with a time limit, and close them before uploading.
- A green LED is not an error to debug in the code. The firmware isn't running;
  upload it.

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

## Computer side

```bash
pip install pyserial matplotlib
python3 scripts/dashboard.py              # live graphs + interpretation
python3 scripts/dashboard.py --demo       # simulated data, no board needed
python3 scripts/dashboard.py --log s.csv  # also save every sample
python3 scripts/read_sensor.py            # plain terminal output
```

On Windows, use `python` instead of `python3`. The scripts find the QT Py by its
USB ID. If that fails or several boards are connected, pass the port, e.g.
`--port COM5` or `--port /dev/ttyACM0`.

Only one program can use the serial port at a time, so close the PlatformIO serial
monitor first, and close the dashboard before uploading firmware.
