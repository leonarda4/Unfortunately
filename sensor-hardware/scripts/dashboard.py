"""Live dashboard: sensor graphs on the left, interpretation on the right.

    python3 scripts/dashboard.py              # connect to the QT Py
    python3 scripts/dashboard.py --log s.csv  # also save every sample to CSV
    python3 scripts/dashboard.py --demo       # simulated data, no board needed

Press r in the window to restart the baseline of the current session.
"""

import argparse
import collections
import csv
import json
import math
import queue
import random
import threading
import time

import matplotlib.pyplot as plt
import serial
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import blended_transform_factory

from read_sensor import FIELDS, find_port

# Same thresholds as the firmware, used for the meter scale and the demo
LEVEL_PCT = 10.0
TREND_PCT_PER_MIN = 10.0
METER_RANGE_PCT = 30.0
HR_MIN_CONFIDENCE = 90
GSR_CONTACT_MARGIN = 30

# Colours
SURFACE   = "#fcfcfb"
PAGE      = "#f9f9f7"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
AXIS      = "#c3c2b7"
NEUTRAL   = "#f0efec"
SERIES    = "#2a78d6"  # the measured signal
SPIKE     = "#eb6834"  # GSR spike markers
CALMER    = "#2a78d6"  # diverging pair: calmer <-> aroused
AROUSED   = "#e34948"

LEVEL_TEXT = {
    "aroused": ("AROUSED", AROUSED, "white"),
    "neutral": ("NEUTRAL", NEUTRAL, INK),
    "calmer":  ("CALMER", CALMER, "white"),
}
TREND_TEXT = {
    "stressing": "↑  Getting more stressed",
    "steady":    "→  Steady",
    "relaxing":  "↓  Relaxing",
}

plt.rcParams.update({
    "font.family": ["Arial", "sans-serif"],  # the macOS Helvetica files only expose regular weight
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "keymap.home": ["h", "home"],  # free up "r" for restarting the baseline
})


# ---------------------------------------------------------------------------
# Data sources: both put parsed JSON messages into a queue
# ---------------------------------------------------------------------------
def read_serial(port, messages):
    while True:
        line = port.readline().decode(errors="ignore").strip()
        if not line:
            continue
        try:
            messages.put(json.loads(line))
        except json.JSONDecodeError:
            messages.put({"event": "unreadable", "line": line})  # partial line, or old firmware


def simulate():
    """Endless fake sessions shaped like the firmware's output."""
    t = 0.0
    while True:
        for _ in range(30):  # a few seconds without a hand
            t += 0.1
            yield {"t": int(t * 1000), "phase": "idle", "hand": 0, "session_s": None, "finger": 0,
                   "hr": 0, "hr_conf": 0, "spo2": 0, "gsr_raw": 700, "gsr_open": 700.0, "gsr": None,
                   "gsr_base": None, "gsr_change": None, "gsr_trend": None, "gsr_phasic": None,
                   "hr_base": None, "hr_change": None, "spikes": 0, "spike": 0, "level": None, "trend": None}

        yield {"event": "session_start", "t": int(t * 1000)}
        g0, hr0 = random.uniform(160, 240), random.uniform(64, 76)
        base = hr_base = None
        history = collections.deque(maxlen=100)
        phasic, spikes, last_spike = 0.0, 0, -99.0

        for i in range(900):
            s = i / 10
            t += 0.1
            stress = min(1, max(0, (s - 35) / 25)) if s < 70 else max(0, 1 - (s - 70) / 15)
            if s > 5 and random.random() < 0.01:
                phasic += g0 * 0.08
                if base:
                    spikes += 1
                    last_spike = s
            phasic *= 0.93
            tonic = g0 * (1 + 0.3 * stress) + random.gauss(0, 0.6) + phasic * 0.4
            history.append(tonic)
            hr = hr0 + 14 * stress + random.gauss(0, 1.2)
            conf = 95 if s > 4 else 40

            if base is None and s >= 15:
                base, hr_base = g0, hr0
                yield {"event": "baseline_done", "t": int(t * 1000), "gsr_base": base, "hr_base": hr_base}

            msg = {"t": int(t * 1000), "phase": "measuring" if base else "baseline", "hand": 1,
                   "session_s": s, "finger": 3, "hr": int(hr), "hr_conf": conf, "spo2": 98 if random.random() < 0.9 else 97,
                   "gsr_raw": int(700 - tonic), "gsr_open": 700.0, "gsr": tonic, "gsr_base": base,
                   "gsr_change": None, "gsr_trend": None, "gsr_phasic": None, "hr_base": hr_base,
                   "hr_change": None, "spikes": spikes, "spike": int(s - last_spike < 3),
                   "level": None, "trend": None}
            if base:
                change = (tonic - base) / base * 100
                trend = (tonic - history[0]) / base * 100 * 6 if len(history) == 100 else None
                msg.update(gsr_change=change, gsr_trend=trend, hr_change=hr - hr_base,
                           gsr_phasic=phasic / base * 100)
                msg["level"] = ("aroused" if change > LEVEL_PCT or hr - hr_base > 10
                                else "calmer" if change < -LEVEL_PCT else "neutral")
                if trend is not None:
                    msg["trend"] = ("stressing" if trend > TREND_PCT_PER_MIN
                                    else "relaxing" if trend < -TREND_PCT_PER_MIN else "steady")
            yield msg

        yield {"event": "session_end", "t": int(t * 1000), "duration_s": 90.0, "gsr_base": base,
               "gsr_change": msg["gsr_change"], "hr_base": hr_base, "hr_change": msg["hr_change"],
               "spikes": spikes}


def run_simulation(messages):
    for msg in simulate():
        messages.put(msg)
        if "event" not in msg:
            time.sleep(0.1)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def style_chart(ax, title, unit):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=8)
    ax.set_ylabel(unit)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)


def finite(values):
    return [v for v in values if v is not None and not math.isnan(v)]


class Dashboard:
    def __init__(self, send_command=None, log_path=None):
        self.send_command = send_command
        self.log_file = open(log_path, "w", newline="") if log_path else None
        self.writer = csv.DictWriter(self.log_file, FIELDS, extrasaction="ignore") if self.log_file else None
        if self.writer:
            self.writer.writeheader()

        self.latest = None
        self.unreadable = 0
        self.summary = None
        self.error = None
        self.reset_session()
        self.build_figure()

    def reset_session(self):
        self.x, self.hr, self.spo2, self.gsr = [], [], [], []
        self.spike_x, self.spike_y = [], []
        self.spikes_seen = 0
        self.baseline_end = None
        self.gsr_base = self.hr_base = None

    # -- figure layout -------------------------------------------------------
    def build_figure(self):
        self.fig = plt.figure(figsize=(13, 7.5), facecolor=PAGE)
        self.fig.canvas.manager.set_window_title("Sensor dashboard")
        grid = self.fig.add_gridspec(3, 2, width_ratios=[2.1, 1], hspace=0.55, wspace=0.16,
                                     left=0.06, right=0.98, top=0.87, bottom=0.08)

        self.ax_hr = self.fig.add_subplot(grid[0, 0])
        self.ax_spo2 = self.fig.add_subplot(grid[1, 0], sharex=self.ax_hr)
        self.ax_gsr = self.fig.add_subplot(grid[2, 0], sharex=self.ax_hr)
        style_chart(self.ax_hr, "Heart rate", "bpm")
        style_chart(self.ax_spo2, "Blood oxygen (SpO$_2$)", "%")
        style_chart(self.ax_gsr, "Skin conductance (GSR)", "level  ↑ more sweat")
        self.ax_gsr.set_xlabel("seconds since hand placed", color=MUTED)

        line_style = dict(color=SERIES, linewidth=1.8, solid_capstyle="round")
        (self.hr_line,) = self.ax_hr.plot([], [], **line_style)
        (self.spo2_line,) = self.ax_spo2.plot([], [], **line_style)
        (self.gsr_line,) = self.ax_gsr.plot([], [], label="GSR level", **line_style)
        (self.spike_dots,) = self.ax_gsr.plot([], [], "o", color=SPIKE, markersize=7,
                                              markeredgecolor=SURFACE, markeredgewidth=1.5, label="spike")
        self.ax_gsr.legend(loc="upper left", frameon=False, ncol=2, fontsize=9, labelcolor=INK_2)

        # Baseline window shading and reference lines
        self.baseline_spans = []
        for ax in (self.ax_hr, self.ax_spo2, self.ax_gsr):
            span = Rectangle((0, 0), 0, 1, transform=blended_transform_factory(ax.transData, ax.transAxes),
                             facecolor=NEUTRAL, edgecolor="none", zorder=0)
            ax.add_patch(span)
            self.baseline_spans.append(span)
        self.baseline_label = self.ax_gsr.text(0.006, 0.04, "", transform=self.ax_gsr.transAxes,
                                               va="bottom", ha="left", fontsize=8, color=MUTED)
        self.hr_ref = self.ax_hr.axhline(float("nan"), color=MUTED, linewidth=1)
        self.gsr_ref = self.ax_gsr.axhline(float("nan"), color=MUTED, linewidth=1)
        self.hr_ref_label = self.ax_hr.text(1.01, 0, "", transform=blended_transform_factory(
            self.ax_hr.transAxes, self.ax_hr.transData), ha="left", va="center", fontsize=8, color=MUTED)
        self.gsr_ref_label = self.ax_gsr.text(1.01, 0, "", transform=blended_transform_factory(
            self.ax_gsr.transAxes, self.ax_gsr.transData), ha="left", va="center", fontsize=8, color=MUTED)

        self.fig.text(0.06, 0.95, "Sensor dashboard", fontsize=16, fontweight="bold", va="center")
        self.header = self.fig.text(0.98, 0.95, "", fontsize=11, color=INK_2, ha="right", va="center")

        self.build_panel(grid[:, 1])

    def build_panel(self, cell):
        ax = self.panel = self.fig.add_subplot(cell)
        ax.set_facecolor(SURFACE)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        def caption(y, text):
            return ax.text(0.07, y, text.upper(), fontsize=8, color=MUTED, fontweight="bold")

        caption(0.93, "State (vs. baseline)")
        self.state_box = FancyBboxPatch((0.07, 0.75), 0.86, 0.15, boxstyle="round,pad=0,rounding_size=0.02",
                                        facecolor=NEUTRAL, edgecolor="none", mutation_aspect=0.5)
        ax.add_patch(self.state_box)
        self.state_text = ax.text(0.5, 0.825, "", fontsize=30, fontweight="bold", ha="center", va="center")
        self.state_sub = ax.text(0.07, 0.715, "", fontsize=10, color=INK_2, va="top")

        self.trend_caption = caption(0.63, "")
        self.trend_text = ax.text(0.07, 0.585, "", fontsize=16, fontweight="bold", va="center")
        self.trend_sub = ax.text(0.07, 0.56, "", fontsize=10, color=INK_2, va="top", linespacing=1.5)

        # Arousal meter: GSR change vs. baseline, calmer <- 0 -> aroused
        caption(0.46, "Arousal meter (GSR vs. baseline)")
        self.meter_track = Rectangle((0.07, 0.385), 0.86, 0.04, facecolor=NEUTRAL, edgecolor="none")
        self.meter_fill = Rectangle((0.5, 0.385), 0, 0.04, facecolor=AROUSED, edgecolor="none")
        ax.add_patch(self.meter_track)
        ax.add_patch(self.meter_fill)
        ax.plot([0.5, 0.5], [0.375, 0.435], color=INK_2, linewidth=1.2)
        ax.text(0.07, 0.36, f"calmer  −{METER_RANGE_PCT:.0f}%", fontsize=8, color=MUTED, va="top")
        ax.text(0.5, 0.36, "baseline", fontsize=8, color=MUTED, va="top", ha="center")
        ax.text(0.93, 0.36, f"+{METER_RANGE_PCT:.0f}%  aroused", fontsize=8, color=MUTED, va="top", ha="right")

        # Four stat tiles
        self.tiles = {}
        for key, label, x, y in (("hr", "Heart rate", 0.07, 0.22), ("spo2", "SpO$_2$", 0.52, 0.22),
                                 ("gsr", "GSR change", 0.07, 0.07), ("spikes", "Spikes", 0.52, 0.07)):
            ax.text(x, y + 0.075, label.upper(), fontsize=8, color=MUTED, fontweight="bold")
            value = ax.text(x, y + 0.03, "–", fontsize=20, fontweight="bold", va="center")
            sub = ax.text(x, y - 0.015, "", fontsize=9, color=INK_2, va="center")
            self.tiles[key] = (value, sub)

        hint = "press r to restart the baseline" if self.send_command else ""
        ax.text(0.93, 0.015, hint, fontsize=8, color=MUTED, ha="right")

    # -- incoming data -------------------------------------------------------
    def handle(self, msg):
        event = msg.get("event")
        if event == "session_start":
            self.reset_session()
            self.summary = None
        elif event == "baseline_done":
            self.baseline_end = self.x[-1] if self.x else 0
            self.gsr_base, self.hr_base = msg.get("gsr_base"), msg.get("hr_base")
        elif event == "session_end":
            self.summary = msg
        elif event == "error":
            self.error = msg.get("msg")
        elif event == "unreadable":
            self.unreadable += 1
        elif event is None:
            self.latest = msg
            if self.writer:
                self.writer.writerow(msg)
            if msg["phase"] != "idle" and msg["session_s"] is not None:
                reliable_hr = msg["finger"] == 3 and msg["hr_conf"] >= HR_MIN_CONFIDENCE and msg["hr"] > 0
                self.x.append(msg["session_s"])
                self.hr.append(msg["hr"] if reliable_hr else math.nan)
                self.spo2.append(msg["spo2"] if reliable_hr and msg["spo2"] > 0 else math.nan)
                self.gsr.append(msg["gsr"] if msg["gsr"] is not None else math.nan)
                if msg["spikes"] > self.spikes_seen:
                    self.spikes_seen = msg["spikes"]
                    self.spike_x.append(msg["session_s"])
                    self.spike_y.append(msg["gsr"])

    def on_key(self, event):
        if event.key == "r" and self.send_command:
            self.send_command(b"r")

    # -- drawing -------------------------------------------------------------
    def draw(self):
        self.draw_charts()
        self.draw_panel()

    def draw_charts(self):
        self.hr_line.set_data(self.x, self.hr)
        self.spo2_line.set_data(self.x, self.spo2)
        self.gsr_line.set_data(self.x, self.gsr)
        self.spike_dots.set_data(self.spike_x, self.spike_y)

        end = self.x[-1] if self.x else 0
        self.ax_hr.set_xlim(0, max(30, end * 1.05))

        for ax, values, pad, ceiling, empty in ((self.ax_hr, self.hr + [self.hr_base], 5, None, (50, 110)),
                                                (self.ax_spo2, self.spo2, 1, 100, (90, 100)),
                                                (self.ax_gsr, self.gsr + [self.gsr_base], 10, None, (0, 100))):
            values = finite(values)
            if values:
                low, high = min(values) - pad, max(values) + pad
                ax.set_ylim(low, min(high, ceiling) if ceiling else high)
            else:
                ax.set_ylim(*empty)

        # Grey band while the baseline is being measured
        baseline_width = self.baseline_end if self.baseline_end is not None else end
        for span in self.baseline_spans:
            span.set_width(baseline_width if self.x else 0)
        self.baseline_label.set_text("baseline window" if self.x else "")

        for ref, label, value, unit in ((self.hr_ref, self.hr_ref_label, self.hr_base, " bpm"),
                                        (self.gsr_ref, self.gsr_ref_label, self.gsr_base, "")):
            y = value if value is not None else math.nan
            ref.set_ydata([y, y])
            label.set_y(y if value is not None else 0)
            label.set_text(f"baseline\n{value:.0f}{unit}" if value is not None else "")

    def draw_panel(self):
        msg = self.latest or {}
        phase = msg.get("phase")

        if self.error:
            self.header.set_text(f"Error: {self.error}")
        elif self.latest is None and self.unreadable > 20:
            self.header.set_text("Receiving data, but not JSON \u2013 upload the new firmware")
        elif phase == "measuring":
            self.header.set_text(f"Measuring · {msg['session_s']:.0f} s")
        elif phase == "baseline":
            self.header.set_text(f"Measuring baseline · keep still · {msg['session_s']:.0f} s")
        elif phase == "idle":
            self.header.set_text("Waiting for a hand on both sensors" + (" · showing last session" if self.x else ""))
        else:
            self.header.set_text("Waiting for data…")

        # State box
        level = msg.get("level")
        if phase == "measuring" and level in LEVEL_TEXT:
            word, box, ink = LEVEL_TEXT[level]
            self.state_sub.set_text(self.state_explanation(msg))
        elif phase == "baseline":
            word, box, ink = "BASELINE…", NEUTRAL, INK_2
            self.state_sub.set_text("Learning this person's resting level.")
        else:
            word, box, ink = "NO HAND", NEUTRAL, MUTED
            self.state_sub.set_text(self.summary_text() or "Place fingers on the GSR and pulse sensor.")
        self.state_text.set_text(word)
        self.state_text.set_color(ink)
        self.state_box.set_facecolor(box)

        # Trend (while idle this area shows what each sensor detects instead)
        trend, value = msg.get("trend"), msg.get("gsr_trend")
        self.trend_caption.set_text("SENSOR CHECK" if phase == "idle" else "TREND (LAST 10 S)")
        self.trend_sub.set_y(0.605 if phase == "idle" else 0.56)
        if phase == "idle":
            self.trend_text.set_text("")
            self.trend_sub.set_text(self.sensor_check(msg))
        elif phase == "measuring" and trend:
            self.trend_text.set_text(TREND_TEXT[trend])
            self.trend_text.set_color(AROUSED if trend == "stressing" else CALMER if trend == "relaxing" else INK)
            self.trend_sub.set_text(f"GSR changing {value:+.1f} %/min")
        else:
            self.trend_text.set_text("–")
            self.trend_text.set_color(MUTED)
            self.trend_sub.set_text("Available 10 s after the baseline." if phase in ("baseline", "measuring") else "")

        # Meter
        change = msg.get("gsr_change") if phase == "measuring" else None
        if change is None:
            self.meter_fill.set_width(0)
        else:
            clipped = max(-METER_RANGE_PCT, min(METER_RANGE_PCT, change))
            width = clipped / METER_RANGE_PCT * 0.43
            self.meter_fill.set_x(0.5 + min(0, width))
            self.meter_fill.set_width(abs(width))
            self.meter_fill.set_facecolor(AROUSED if change > 0 else CALMER)

        # Tiles
        in_session = phase in ("baseline", "measuring")
        reliable = in_session and msg.get("hr_conf", 0) >= HR_MIN_CONFIDENCE and msg.get("hr", 0) > 0
        hr_change = msg.get("hr_change")
        self.set_tile("hr", f"{msg['hr']} bpm" if reliable else "–",
                      f"{hr_change:+.0f} vs. baseline" if reliable and hr_change is not None
                      else "low confidence" if in_session else "")
        self.set_tile("spo2", f"{msg['spo2']} %" if reliable and msg.get("spo2") else "–", "")
        self.set_tile("gsr", f"{change:+.1f} %" if change is not None else "–",
                      "vs. baseline" if change is not None else "")
        spikes = msg.get("spikes", 0) if in_session else 0
        self.set_tile("spikes", str(spikes) if phase == "measuring" else "–",
                      "just now!" if msg.get("spike") else "this session" if phase == "measuring" else "")
        self.tiles["spikes"][1].set_color(SPIKE if msg.get("spike") else INK_2)

    def set_tile(self, key, value, sub):
        self.tiles[key][0].set_text(value)
        self.tiles[key][1].set_text(sub)

    def sensor_check(self, msg):
        finger = msg["finger"] == 3
        contact = msg["gsr_open"] - msg["gsr_raw"] > GSR_CONTACT_MARGIN
        return (f"Pulse sensor: {'finger detected' if finger else 'no finger'} (status {msg['finger']})\n"
                f"GSR: {'skin contact' if contact else 'no contact'} (raw {msg['gsr_raw']}, "
                f"open {msg['gsr_open']:.0f})\n"
                f"A session starts when both detect a hand.")

    def state_explanation(self, msg):
        parts = []
        if msg.get("gsr_change") is not None:
            parts.append(f"GSR {msg['gsr_change']:+.0f}%")
        if msg.get("hr_change") is not None:
            parts.append(f"heart rate {msg['hr_change']:+.0f} bpm")
        return "  ·  ".join(parts) + " vs. baseline"

    def summary_text(self):
        s = self.summary
        if not s:
            return None
        parts = [f"Last session: {s['duration_s']:.0f} s"]
        if s.get("gsr_change") is not None:
            parts.append(f"GSR {s['gsr_change']:+.0f}%")
        if s.get("hr_change") is not None:
            parts.append(f"HR {s['hr_change']:+.0f} bpm")
        parts.append(f"{s['spikes']} spikes")
        return ", ".join(parts)

    def close(self):
        if self.log_file:
            self.log_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--log", default=None, help="CSV file to write every sample to")
    parser.add_argument("--demo", action="store_true", help="simulated data, no board needed")
    args = parser.parse_args()

    messages = queue.Queue()

    if args.demo:
        send = None
        threading.Thread(target=run_simulation, args=(messages,), daemon=True).start()
    else:
        port = serial.Serial(args.port or find_port(), 115200, timeout=1)
        send = port.write
        threading.Thread(target=read_serial, args=(port, messages), daemon=True).start()

    dashboard = Dashboard(send_command=send, log_path=args.log)
    dashboard.fig.canvas.mpl_connect("key_press_event", dashboard.on_key)

    def update(_frame):
        while not messages.empty():
            dashboard.handle(messages.get())
        dashboard.draw()

    _animation = FuncAnimation(dashboard.fig, update, interval=200, cache_frame_data=False)
    try:
        plt.show()
    finally:
        dashboard.close()


if __name__ == "__main__":
    main()
