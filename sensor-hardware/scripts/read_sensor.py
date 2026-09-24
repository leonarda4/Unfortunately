"""Read the sensor's JSON lines from the QT Py and optionally log them to CSV.

    pip install pyserial
    python scripts/read_sensor.py                 # print live values
    python scripts/read_sensor.py --log out.csv   # also save every sample

Type r + Enter while running to restart the baseline of the current session.
"""

import argparse
import csv
import glob
import json
import sys
import threading

import serial

FIELDS = [
    "t", "phase", "hand", "session_s", "finger", "hr", "hr_conf", "spo2",
    "gsr_raw", "gsr_open", "gsr", "gsr_base", "gsr_change", "gsr_trend",
    "gsr_phasic", "hr_base", "hr_change", "spikes", "spike", "level", "trend",
]


def find_port():
    ports = glob.glob("/dev/cu.usbmodem*")
    if not ports:
        sys.exit("No QT Py found (no /dev/cu.usbmodem* port).")
    return ports[0]


def forward_commands(port):
    for line in sys.stdin:
        port.write(line.strip().encode())


def fmt(value, unit=""):
    return "-" if value is None else f"{value:+.1f}{unit}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--log", default=None, help="CSV file to write every sample to")
    args = parser.parse_args()

    port = serial.Serial(args.port or find_port(), 115200, timeout=1)
    threading.Thread(target=forward_commands, args=(port,), daemon=True).start()

    log_file = open(args.log, "w", newline="") if args.log else None
    writer = csv.DictWriter(log_file, FIELDS, extrasaction="ignore") if log_file else None
    if writer:
        writer.writeheader()

    try:
        while True:
            line = port.readline().decode(errors="ignore").strip()
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial line, e.g. right after connecting

            if "event" in msg:
                print(f"--- {msg}")
                continue

            if writer:
                writer.writerow(msg)

            if msg["phase"] == "measuring":
                print(
                    f"HR {msg['hr']:3d} ({fmt(msg['hr_change'], ' bpm')})  "
                    f"SpO2 {msg['spo2']:3d}%  "
                    f"GSR {fmt(msg['gsr_change'], '%')} trend {fmt(msg['gsr_trend'], '%/min')}  "
                    f"{msg['level']:>7} / {msg['trend'] or '-':<9} "
                    f"spikes {msg['spikes']}{'  SPIKE!' if msg['spike'] else ''}"
                )
            elif msg["phase"] == "baseline":
                print(f"measuring baseline... {msg['session_s']:.0f} s  HR {msg['hr']}  GSR raw {msg['gsr_raw']}")
            else:
                print(f"waiting for hand  (finger {msg['finger']}, GSR raw {msg['gsr_raw']}, open {msg['gsr_open']:.0f})")
    except KeyboardInterrupt:
        pass
    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
