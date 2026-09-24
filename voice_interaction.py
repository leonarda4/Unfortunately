"""Simple spoken-prompt and response analyzer.

The microphone signal is used for timing and pause metrics; Whisper is used
only for transcription. Press q to quit, r to repeat the prompt, or space to
force-finish a response.
"""

import argparse
import math
import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel


SAMPLE_RATE = 16_000
BLOCK_SIZE = 512
PROMPT = "Tell me about something that made you feel good today."
FILLED_PAUSES = {"um", "uh", "er", "erm", "hmm", "mm", "like"}


@dataclass
class Response:
    audio: np.ndarray
    levels: list[float]
    speech_flags: list[bool]
    started_at: float
    ended_at: float


@dataclass
class Dashboard:
    state: str = "Starting"
    prompt: str = PROMPT
    transcript: str = ""
    response_latency: Optional[float] = None
    response_duration: Optional[float] = None
    speech_duration: Optional[float] = None
    silence_duration: Optional[float] = None
    pause_count: int = 0
    mean_pause: Optional[float] = None
    longest_pause: Optional[float] = None
    words_per_minute: Optional[float] = None
    word_count: int = 0
    filled_pauses: int = 0
    mean_db: Optional[float] = None
    peak_db: Optional[float] = None
    speech_ratio: Optional[float] = None
    noise_floor: float = -55.0
    threshold: float = -45.0
    error: str = ""


def dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
    return 20.0 * math.log10(max(rms, 1e-7))


def format_value(value, suffix="--"):
    return suffix if value is None else f"{value:.1f}"


def speak(text: str) -> None:
    subprocess.run(["say", text], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def compute_metrics(response: Response, transcript: str, dashboard: Dashboard) -> None:
    block_seconds = BLOCK_SIZE / SAMPLE_RATE
    levels = np.asarray(response.levels)
    flags = np.asarray(response.speech_flags, dtype=bool)
    pauses = []
    silence_blocks = 0
    for flag in flags:
        if flag:
            if silence_blocks:
                pauses.append(silence_blocks * block_seconds)
                silence_blocks = 0
        else:
            silence_blocks += 1
    if silence_blocks:
        pauses.append(silence_blocks * block_seconds)

    words = transcript.split()
    duration = max(response.ended_at - response.started_at, block_seconds)
    speech_duration = float(flags.sum() * block_seconds)
    internal_pauses = [pause for pause in pauses if pause >= 0.25]
    filled = sum(word.lower().strip(".,!?;:") in FILLED_PAUSES for word in words)

    dashboard.transcript = transcript or "(no speech recognized)"
    dashboard.response_duration = duration
    dashboard.speech_duration = speech_duration
    dashboard.silence_duration = max(0.0, duration - speech_duration)
    dashboard.pause_count = len(internal_pauses)
    dashboard.mean_pause = float(np.mean(internal_pauses)) if internal_pauses else None
    dashboard.longest_pause = max(internal_pauses, default=None)
    dashboard.word_count = len(words)
    dashboard.words_per_minute = len(words) / max(speech_duration / 60.0, 1 / 60.0)
    dashboard.filled_pauses = filled
    dashboard.mean_db = float(np.mean(levels)) if len(levels) else None
    dashboard.peak_db = float(np.max(levels)) if len(levels) else None
    dashboard.speech_ratio = speech_duration / duration


def transcribe(response: Response, model: WhisperModel, dashboard: Dashboard) -> None:
    dashboard.state = "Transcribing"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_path = Path(audio_file.name)
    try:
        sf.write(audio_path, response.audio, SAMPLE_RATE)
        segments, _ = model.transcribe(str(audio_path), beam_size=1, vad_filter=False)
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        compute_metrics(response, transcript, dashboard)
        dashboard.state = "Complete - press r"
    except Exception as error:
        dashboard.error = f"Transcription failed: {error}"
        dashboard.state = "Error - press r"
    finally:
        audio_path.unlink(missing_ok=True)


def draw_dashboard(dashboard: Dashboard, elapsed: float) -> np.ndarray:
    canvas = np.full((720, 1040, 3), (24, 28, 34), dtype=np.uint8)
    cv2.putText(canvas, "VOICE RESPONSE LAB", (35, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 220, 255), 2)
    cv2.putText(canvas, dashboard.state, (700, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 240, 150), 2)
    cv2.putText(canvas, f"Prompt: {dashboard.prompt}", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (225, 225, 225), 1)

    rows = [
        ("Response latency", f"{format_value(dashboard.response_latency)} s"),
        ("Response duration", f"{format_value(dashboard.response_duration)} s"),
        ("Speech duration", f"{format_value(dashboard.speech_duration)} s"),
        ("Silence duration", f"{format_value(dashboard.silence_duration)} s"),
        ("Pause count (>= 250 ms)", str(dashboard.pause_count)),
        ("Mean / longest pause", f"{format_value(dashboard.mean_pause)} / {format_value(dashboard.longest_pause)} s"),
        ("Words / WPM", f"{dashboard.word_count} / {format_value(dashboard.words_per_minute)}"),
        ("Filled pauses", str(dashboard.filled_pauses)),
        ("Mean / peak level", f"{format_value(dashboard.mean_db)} / {format_value(dashboard.peak_db)} dBFS"),
        ("Speech ratio", f"{format_value(None if dashboard.speech_ratio is None else dashboard.speech_ratio * 100)} %"),
        ("Noise / threshold", f"{dashboard.noise_floor:.1f} / {dashboard.threshold:.1f} dBFS"),
    ]
    for index, (label, value) in enumerate(rows):
        y = 145 + index * 34
        cv2.putText(canvas, label, (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (150, 165, 180), 1)
        cv2.putText(canvas, value, (370, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1)

    cv2.putText(canvas, "Transcript", (650, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 255), 1)
    transcript = dashboard.transcript or "Waiting for a response..."
    for index, line_start in enumerate(range(0, len(transcript), 42)):
        cv2.putText(canvas, transcript[line_start:line_start + 42], (650, 285 + index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)
    if dashboard.error:
        cv2.putText(canvas, dashboard.error[:60], (35, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 100, 255), 1)
    cv2.putText(canvas, "q quit   r repeat prompt   space finish response", (35, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 170, 180), 1)
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description="Speak a prompt, listen for a response, and show speech metrics.")
    parser.add_argument("--model", default="tiny.en", help="faster-whisper model, e.g. tiny.en or base.en")
    parser.add_argument("--prompt", default=PROMPT, help="Prompt spoken to the user")
    parser.add_argument("--silence", type=float, default=1.0, help="Seconds of silence that end a response")
    parser.add_argument("--max-seconds", type=float, default=60.0, help="Maximum response length")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="Whisper device")
    return parser.parse_args()


def main():
    args = parse_args()
    dashboard = Dashboard(prompt=args.prompt)
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, callback_time, status):
        if status:
            dashboard.error = str(status)
        audio_queue.put(indata[:, 0].copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, dtype="float32", callback=audio_callback)
    dashboard.state = "Loading Whisper"
    model = WhisperModel(args.model, device=args.device, compute_type="int8" if args.device == "cpu" else "float16")
    stream.start()

    recording = []
    levels = []
    flags = []
    noise_samples = []
    response_started = None
    prompt_finished = None
    force_finish = False

    def start_prompt():
        nonlocal prompt_finished
        speak(args.prompt)
        prompt_finished = time.monotonic()
        dashboard.state = "Listening"

    threading.Thread(target=start_prompt, daemon=True).start()
    try:
        while True:
            now = time.monotonic()
            while True:
                try:
                    block = audio_queue.get_nowait()
                except queue.Empty:
                    break
                level = dbfs(block)
                if prompt_finished is None:
                    continue
                if response_started is None:
                    noise_samples.append(level)
                    dashboard.noise_floor = float(np.median(noise_samples[-40:]))
                    dashboard.threshold = max(-45.0, dashboard.noise_floor + 10.0)
                    is_speech = level > dashboard.threshold
                    if is_speech:
                        response_started = now
                        dashboard.response_latency = response_started - prompt_finished
                else:
                    is_speech = level > dashboard.threshold
                recording.append(block)
                levels.append(level)
                flags.append(is_speech)

            if response_started is not None:
                last_speech_index = next((index for index in range(len(flags) - 1, -1, -1) if flags[index]), None)
                last_speech = response_started + (last_speech_index + 1) * (BLOCK_SIZE / SAMPLE_RATE) if last_speech_index is not None else response_started
                if force_finish or now - last_speech >= args.silence or now - response_started >= args.max_seconds:
                    response = Response(np.concatenate(recording), levels, flags, response_started, now)
                    dashboard.state = "Transcribing"
                    threading.Thread(target=transcribe, args=(response, model, dashboard), daemon=True).start()
                    response_started = None
                    recording, levels, flags = [], [], []
                    force_finish = False

            frame = draw_dashboard(dashboard, 0.0 if response_started is None else now - response_started)
            cv2.imshow("Voice Interaction", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and response_started is None and dashboard.state != "Transcribing":
                dashboard.transcript = ""
                dashboard.error = ""
                dashboard.state = "Prompting"
                prompt_finished = None
                threading.Thread(target=start_prompt, daemon=True).start()
            if key == 32 and response_started is not None:
                force_finish = True
    finally:
        stream.stop()
        stream.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()