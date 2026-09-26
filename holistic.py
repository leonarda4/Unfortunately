"""Live MediaPipe Holistic face, pose, and hand landmark preview."""

import argparse
from datetime import datetime, timezone
import queue
import random
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
import glob
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import serial
import sounddevice as sd
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections

from voice_interaction import BLOCK_SIZE, SAMPLE_RATE, Dashboard, Response, compute_metrics, dbfs, transcribe


DEFAULT_MODEL = Path(__file__).with_name("models") / "holistic_landmarker.task"
EMOTION_INTERVAL = 15
BLINK_THRESHOLD = 0.1
BLINK_MIN_FRAMES = 1
SENSOR_BAUD = 115200
GRAPH_SAMPLES = 120
GRAPH_SAMPLE_INTERVAL = 0.5
CALIBRATION_MUSIC_SAMPLE_RATE = 22050
FONT_PATHS = (
    Path("/System/Library/Fonts/Helvetica.ttc"),
    Path("/System/Library/Fonts/HelveticaNeue.ttc"),
)
GREETINGS = (
    "Hi! Welcome to the screening, please stand on the marked spot.",
    "Welcome, please stand on the marking on the floor.",
    "Welcome. Please step onto the floor marking to begin.",
    "Hello. Please position yourself on the marked spot.",
)
MARK_PROMPT = "Please stand on the marked spot."
SCAN_ACTIONS = (
    ("right_hand", "Please lift your right hand."),
    ("left_hand", "Please lift your left hand.")
)
TURN_LEFT_PROMPT = "Turn to the left."
TURN_AROUND_PROMPT = "Turn around."
POSITION_PROMPT = "Please come forward and place your right hand on the sensor."
HEAD_PROMPT = "Please lower your head so you see yourself in the mirror."
HAND_PROMPT = "Hand is not detected. Please place three fingers on the sensors."
NO_ANSWER_PROMPTS = (
    "Please answer the question.",
    "Are you going to say something?",
    "Waiting for answer.",
)
LAST_CHANCE_PROMPTS = (
    "You are not going to answer, huh? This is your last chance.",
    "Are you ignoring me?",
)
END_INTERVIEW_PROMPT = (
    "Unfortunately, you need to be able to speak for this role. We will be moving forward with other candidates."
)
POSE_RETRY_SECONDS = 4.0
POSE_STABLE_FRAMES = 3

BASIC_QUESTIONS = (
    "What is your height?",
    "How many siblings do you have?",
    "Are you left-handed or right-handed?",
    "How long did it take you to arrive here today?",
)
ABSURD_QUESTIONS = (
    "Exactly how many steps did you make today?",
    "How long is your right arm?",
    "Which animal would you say you are?",
)
COGNITIVE_QUESTIONS = (
    "What is 9 x 20?",
    "What is 7 + 26?",
    "Name 3 objects that are commonly found in an office.",
    "Repeat the following sequence: 7, 2, 9, 4, 5.",
    "Why did the chicken cross the road?",
    "A pen and a paper cost €1.10 in total. The pen costs $1.00 more than the paper. How much does the paper cost?",
)
WORKPLACE_QUESTIONS = (
    "Do you prefer working alone or as part of a team?",
    "What do you do when instructions are unclear?",
    "Would you challenge a decision made by your supervisor?",
    "Describe the last mistake you made.",
    "Have you ever concealed a mistake at work?",
    "Would you complete a task you believed was unnecessary?",
    "Is following procedure more important than achieving the desired result?",
    "How long should an employer tolerate poor performance?",
    "Would your previous employer hire you again?",
    "Which is more important: speed or accuracy?",
)
EMOTIONAL_QUESTIONS = (
    "Are both of your parents alive?",
    "Has a personal relationship ever affected your work?",
    "When did you last feel afraid?",
    "Do you believe stress improves your performance?",
    "Are you currently trying to control your expression?",
    "Have you changed any answer to appear more employable?",
    "Would you lie to obtain this position?",
)
HEALTH_QUESTIONS = (
    "Did you eat breakfast today?",
    "How many hours did you sleep last night?",
    "What is your favourite meal?",
    "How much protein do you eat daily?",
    "Our mandatory team-building events involve an outdoor marathon. Is there anything in your family's medical history that suggests you might let the team down on the track?",
    "How comfortable are you with a mandatory microchip upgrade?",
    "Are there any chronic medical conditions that run in your family?",
)


def screening_prompt(category, text, parts=None, pause_after=0.0, sound=None):
    return {
        "category": category,
        "text": text,
        "parts": parts or [text],
        "pause_after": pause_after,
        "sound": sound,
    }


def select_followup_questions(gender):
    """Choose the requested number of prompts for each screening category."""
    questions = [screening_prompt("Personal baseline", "How old are you?")]
    questions.append(screening_prompt("Basic", random.choice(BASIC_QUESTIONS)))
    if random.random() < 0.2:
        questions.append(screening_prompt("Absurd", random.choice(ABSURD_QUESTIONS)))
    questions.extend((
        screening_prompt("Cognitive load", random.choice(COGNITIVE_QUESTIONS)),
        screening_prompt("Workplace judgment", random.choice(WORKPLACE_QUESTIONS)),
        screening_prompt("Emotional pressure", random.choice(EMOTIONAL_QUESTIONS)),
        screening_prompt("Health", random.choice(HEALTH_QUESTIONS)),
    ))
    if "woman" in gender.lower() or "female" in gender.lower():
        if random.choice((True, False)):
            questions.append(screening_prompt(
                "Women specific",
                "Close your eyes. Imagine a cute baby laughing.",
                parts=["Close your eyes.", "Imagine a cute baby laughing."],
                pause_after=1.0,
                sound="laugh",
            ))
        else:
            sound = random.choice(("laugh", "cry", "speak"))
            questions.append(screening_prompt(
                "Women specific",
                "Please close your eyes and listen to the following sound.",
                sound=sound,
            ))
    return questions


@dataclass
class VisualMetrics:
    emotion: str = "warming up"
    emotion_confidence: float = 0.0
    emotion_error: str = ""
    looking_at_camera: bool = False
    gaze_available: bool = False
    blink_count: int = 0
    eyes_closed_frames: int = 0


@dataclass
class SensorMetrics:
    connected: bool = False
    phase: str = "offline"
    hr: object = None
    hr_confidence: object = None
    finger: object = None
    spo2: object = None
    gsr: object = None
    gsr_change: object = None
    gsr_trend: object = None
    level: str = "-"
    trend: str = "-"
    spikes: object = None
    error: str = ""


def speak(message):
    """Speak without blocking the camera preview on macOS."""
    subprocess.Popen(
        ["say", message],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def find_sensor_port():
    ports = glob.glob("/dev/cu.usbmodem*")
    return ports[0] if ports else None


def sensor_worker(port_name, metrics, stop_event):
    """Read QT Py JSON samples without blocking the camera loop."""
    try:
        port = serial.Serial(port_name, SENSOR_BAUD, timeout=1)
    except Exception as error:
        metrics.error = f"Sensor unavailable: {error}"
        return

    metrics.connected = True
    metrics.error = ""
    try:
        while not stop_event.is_set():
            line = port.readline().decode(errors="ignore").strip()
            if not line or not line.startswith("{"):
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "event" in message:
                continue
            metrics.phase = message.get("phase", metrics.phase)
            metrics.hr = message.get("hr")
            metrics.hr_confidence = message.get("hr_conf")
            metrics.finger = message.get("finger")
            metrics.spo2 = message.get("spo2")
            metrics.gsr = message.get("gsr")
            metrics.gsr_change = message.get("gsr_change")
            metrics.gsr_trend = message.get("gsr_trend")
            metrics.level = message.get("level") or "-"
            metrics.trend = message.get("trend") or "-"
            metrics.spikes = message.get("spikes")
    except Exception as error:
        metrics.error = f"Sensor read error: {error}"
    finally:
        metrics.connected = False
        port.close()


def landmark_group(collection):
    """Normalize Tasks results that may be flat or wrapped in one face/group."""
    if not collection:
        return []
    first = collection[0]
    return collection if hasattr(first, "x") else first


def right_hand_is_raised(results):
    """Return true when the user's semantic right wrist is above the shoulder."""
    if not results.right_hand_landmarks or not results.pose_landmarks:
        return False

    hand_landmarks = landmark_group(results.right_hand_landmarks)
    pose_landmarks = landmark_group(results.pose_landmarks)
    if not hand_landmarks or not pose_landmarks:
        return False
    wrist = hand_landmarks[0]
    shoulder = pose_landmarks[12]
    return wrist.y < shoulder.y - 0.05


def landmark_is_visible(landmark, threshold=0.5):
    visibility = getattr(landmark, "visibility", None)
    presence = getattr(landmark, "presence", None)
    confidence = visibility if visibility is not None else presence
    if confidence is not None and confidence < threshold:
        return False
    return 0.0 <= landmark.x <= 1.0 and 0.0 <= landmark.y <= 1.0


def hips_are_visible(results):
    pose = landmark_group(results.pose_landmarks)
    return len(pose) > 24 and all(landmark_is_visible(pose[index]) for index in (23, 24))


def scan_pose_is_satisfied(action, results, face_present):
    pose = landmark_group(results.pose_landmarks)
    if not hips_are_visible(results):
        return False

    if action in ("right_hand", "left_hand"):
        hands = results.right_hand_landmarks if action == "right_hand" else results.left_hand_landmarks
        hand = landmark_group(hands)
        shoulder_index = 12 if action == "right_hand" else 11
        if not hand or len(pose) <= shoulder_index:
            return False
        wrist = hand[0]
        shoulder = pose[shoulder_index]
        return landmark_is_visible(wrist) and landmark_is_visible(shoulder) and wrist.y < shoulder.y - 0.05

    if action in ("right_leg", "left_leg"):
        hip_index, ankle_index = (24, 28) if action == "right_leg" else (23, 27)
        return (
            len(pose) > ankle_index
            and landmark_is_visible(pose[ankle_index])
            and pose[ankle_index].y < pose[hip_index].y - 0.08
        )

    if len(pose) <= 24 or not all(landmark_is_visible(pose[index]) for index in (11, 12, 23, 24)):
        return False
    shoulder_width = abs(pose[11].x - pose[12].x)
    hip_width = abs(pose[23].x - pose[24].x)
    if action == "turn_left":
        return shoulder_width < max(hip_width * 0.72, 0.08)
    if action == "turn_around":
        return not face_present
    return False


def no_answer_action(level, silent_for, nudge_after, warning_after, final_after):
    if level == 0 and silent_for >= nudge_after:
        return "reminder"
    if level == 1 and silent_for >= warning_after:
        return "last_chance"
    if level == 2 and silent_for >= final_after:
        return "end_interview"
    return None


def build_calibration_music(sample_rate=CALIBRATION_MUSIC_SAMPLE_RATE):
    """Synthesize a soft, looping lounge-style chord progression."""
    chord_progression = (
        (261.63, 329.63, 392.00, 493.88),
        (174.61, 220.00, 261.63, 329.63),
        (220.00, 261.63, 329.63, 392.00),
        (196.00, 246.94, 293.66, 349.23),
    )
    chord_seconds = 2.0
    samples_per_chord = int(sample_rate * chord_seconds)
    total_samples = samples_per_chord * len(chord_progression)
    music = np.zeros(total_samples, dtype=np.float64)
    time_axis = np.arange(samples_per_chord) / sample_rate

    for chord_index, chord in enumerate(chord_progression):
        start = chord_index * samples_per_chord
        end = start + samples_per_chord
        swell = 0.72 + 0.28 * np.sin(np.pi * time_axis / chord_seconds) ** 2
        pad = np.zeros(samples_per_chord, dtype=np.float64)
        for frequency in chord:
            pad += np.sin(2 * np.pi * frequency * time_axis)
            pad += 0.18 * np.sin(2 * np.pi * frequency * 2 * time_axis)
        music[start:end] += pad * swell * 0.018

        for note_index, frequency in enumerate(chord):
            note_start = note_index * int(sample_rate * 0.45)
            note_length = min(int(sample_rate * 0.7), samples_per_chord - note_start)
            note_time = np.arange(note_length) / sample_rate
            envelope = np.exp(-note_time * 4.2)
            music[start + note_start:start + note_start + note_length] += (
                0.045 * np.sin(2 * np.pi * frequency * 2 * note_time) * envelope
            )

    fade_samples = int(sample_rate * 0.08)
    fade = np.linspace(0.0, 1.0, fade_samples)
    music[:fade_samples] *= fade
    music[-fade_samples:] *= fade[::-1]
    return music.astype(np.float32)


def write_test_log(log_path, entry):
    """Append one structured event to the local session JSONL file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def response_log_entry(session_id, question, dashboard, visual_metrics, sensor_metrics):
    return {
        "event": "response",
        "session_id": session_id,
        "category": question["category"],
        "question": question["text"],
        "transcript": dashboard.transcript,
        "response_latency_seconds": dashboard.response_latency,
        "response_duration_seconds": dashboard.response_duration,
        "speech_duration_seconds": dashboard.speech_duration,
        "silence_duration_seconds": dashboard.silence_duration,
        "pause_count": dashboard.pause_count,
        "mean_pause_seconds": dashboard.mean_pause,
        "longest_pause_seconds": dashboard.longest_pause,
        "word_count": dashboard.word_count,
        "words_per_minute": dashboard.words_per_minute,
        "filled_pauses": dashboard.filled_pauses,
        "mean_dbfs": dashboard.mean_db,
        "peak_dbfs": dashboard.peak_db,
        "speech_ratio": dashboard.speech_ratio,
        "visual": {
            "emotion": visual_metrics.emotion,
            "emotion_confidence": visual_metrics.emotion_confidence,
            "blink_count": visual_metrics.blink_count,
            "looking_at_camera": visual_metrics.looking_at_camera,
            "gaze_available": visual_metrics.gaze_available,
        },
        "sensor": {
            "phase": sensor_metrics.phase,
            "heart_rate": sensor_metrics.hr,
            "heart_rate_confidence": sensor_metrics.hr_confidence,
            "spo2": sensor_metrics.spo2,
            "gsr": sensor_metrics.gsr,
            "gsr_change": sensor_metrics.gsr_change,
            "gsr_trend": sensor_metrics.gsr_trend,
            "level": sensor_metrics.level,
            "trend": sensor_metrics.trend,
            "spikes": sensor_metrics.spikes,
        },
    }


def sensor_calibration_action(phase, connected):
    if not connected:
        return "skip"
    if phase == "baseline":
        return "start"
    if phase == "measuring":
        return "complete"
    if phase == "idle":
        return "retry"
    return "wait"


def eye_aspect_ratio(landmarks, horizontal_left, horizontal_right, vertical_top, vertical_bottom):
    """Estimate eye openness from normalized face landmark coordinates."""
    horizontal = np.linalg.norm(
        np.array([landmarks[horizontal_left].x, landmarks[horizontal_left].y])
        - np.array([landmarks[horizontal_right].x, landmarks[horizontal_right].y])
    )
    vertical = np.linalg.norm(
        np.array([landmarks[vertical_top].x, landmarks[vertical_top].y])
        - np.array([landmarks[vertical_bottom].x, landmarks[vertical_bottom].y])
    )
    return vertical / max(horizontal, 1e-6)


def update_blink_count(face_landmarks, metrics):
    """Count a blink after the eyes remain closed briefly and reopen."""
    left_ear = eye_aspect_ratio(face_landmarks, 33, 133, 159, 145)
    right_ear = eye_aspect_ratio(face_landmarks, 362, 263, 386, 374)
    eyes_closed = (left_ear + right_ear) / 2.0 < BLINK_THRESHOLD
    if eyes_closed:
        metrics.eyes_closed_frames += 1
    elif metrics.eyes_closed_frames >= BLINK_MIN_FRAMES:
        metrics.blink_count += 1
        metrics.eyes_closed_frames = 0
    else:
        metrics.eyes_closed_frames = 0


def gaze_is_camera_facing(face_landmarks):
    """Use iris position within each eye as a coarse camera-gaze proxy."""
    if len(face_landmarks) < 478:
        return False, False

    eye_ranges = ((33, 133, 468), (362, 263, 473))
    horizontal_positions = []
    vertical_positions = []
    for left_corner, right_corner, iris_index in eye_ranges:
        corner_left = face_landmarks[left_corner]
        corner_right = face_landmarks[right_corner]
        iris = face_landmarks[iris_index]
        horizontal_span = corner_right.x - corner_left.x
        if abs(horizontal_span) < 1e-6:
            return False, False
        horizontal_positions.append((iris.x - corner_left.x) / horizontal_span)
        vertical_positions.append(iris.y - (corner_left.y + corner_right.y) / 2.0)

    centered_horizontally = all(0.25 <= position <= 0.75 for position in horizontal_positions)
    centered_vertically = all(abs(position) <= 0.08 for position in vertical_positions)
    return centered_horizontally and centered_vertically, True


def face_crop(frame, face_landmarks):
    """Return a padded face crop for the emotion model."""
    height, width = frame.shape[:2]
    x_values = [landmark.x for landmark in face_landmarks]
    y_values = [landmark.y for landmark in face_landmarks]
    left = max(0, int(min(x_values) * width) - 20)
    top = max(0, int(min(y_values) * height) - 20)
    right = min(width, int(max(x_values) * width) + 20)
    bottom = min(height, int(max(y_values) * height) + 20)
    crop = frame[top:bottom, left:right]
    return crop if crop.size else None


def emotion_worker(emotion_queue, metrics, stop_event):
    """Run DeepFace on the newest crop without blocking landmark rendering."""
    try:
        from deepface import DeepFace
    except ImportError as error:
        metrics.emotion_error = f"DeepFace unavailable: {error}"
        return

    while not stop_event.is_set():
        try:
            crop = emotion_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            result = DeepFace.analyze(
                crop,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip",
                silent=True,
            )
            result = result[0] if isinstance(result, list) else result
            emotions = result.get("emotion", {})
            dominant = result.get("dominant_emotion", "unknown")
            metrics.emotion = str(dominant)
            metrics.emotion_confidence = float(emotions.get(dominant, 0.0))
            metrics.emotion_error = ""
        except Exception as error:
            metrics.emotion_error = f"Emotion error: {error}"


def print_response_metrics(dashboard, visual_metrics, sensor_metrics):
    """Print the completed response in a terminal-friendly format."""
    print("\n--- Voice response ---")
    print(f"Transcription: {dashboard.transcript}")
    print(f"Response latency: {dashboard.response_latency:.2f} s")
    print(f"Response duration: {dashboard.response_duration:.2f} s")
    print(f"Speech duration: {dashboard.speech_duration:.2f} s")
    print(f"Silence duration: {dashboard.silence_duration:.2f} s")
    print(f"Pause count (>= 250 ms): {dashboard.pause_count}")
    print(f"Mean pause: {dashboard.mean_pause or 0.0:.2f} s")
    print(f"Longest pause: {dashboard.longest_pause or 0.0:.2f} s")
    print(f"Words: {dashboard.word_count}")
    print(f"Words per minute: {dashboard.words_per_minute:.1f}")
    print(f"Filled pauses: {dashboard.filled_pauses}")
    print(f"Mean / peak level: {dashboard.mean_db:.1f} / {dashboard.peak_db:.1f} dBFS")
    print(f"Speech ratio: {dashboard.speech_ratio * 100:.1f}%")
    print(f"Dominant emotion: {visual_metrics.emotion} ({visual_metrics.emotion_confidence:.1f}%)")
    print(f"Blink count: {visual_metrics.blink_count}")
    print(f"Looking at camera: {'yes' if visual_metrics.looking_at_camera else 'no'}")
    print(f"Sensor HR / SpO2: {sensor_metrics.hr or '-'} bpm / {sensor_metrics.spo2 or '-'}%")
    print(f"Sensor GSR change / trend: {sensor_metrics.gsr_change or '-'}% / {sensor_metrics.gsr_trend or '-'}%/min")
    print(f"Sensor level / trend: {sensor_metrics.level} / {sensor_metrics.trend}")
    print(f"Sensor spikes: {sensor_metrics.spikes if sensor_metrics.spikes is not None else '-'}")
    print("----------------------\n", flush=True)


def draw_landmarks(frame, results):
    height, width = frame.shape[:2]

    def point(landmark):
        return round(landmark.x * width), round(landmark.y * height)

    def draw_connections(landmarks, connections, color, thickness, radius):
        if not landmarks:
            return
        for connection in connections:
            start = connection.start
            end = connection.end
            if start < len(landmarks) and end < len(landmarks):
                cv2.line(frame, point(landmarks[start]), point(landmarks[end]), color, thickness, cv2.LINE_AA)
        for landmark in landmarks:
            cv2.circle(frame, point(landmark), radius, color, -1, cv2.LINE_AA)

    face_landmarks = landmark_group(results.face_landmarks)
    draw_connections(
        face_landmarks,
        FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        (70, 245, 130),
        1,
        1,
    )
    draw_connections(
        face_landmarks,
        FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        (245, 255, 255),
        2,
        1,
    )
    draw_connections(
        landmark_group(results.pose_landmarks),
        PoseLandmarksConnections.POSE_LANDMARKS,
        (40, 205, 255),
        3,
        3,
    )
    draw_connections(
        landmark_group(results.left_hand_landmarks),
        HandLandmarksConnections.HAND_CONNECTIONS,
        (255, 190, 55),
        2,
        3,
    )
    draw_connections(
        landmark_group(results.right_hand_landmarks),
        HandLandmarksConnections.HAND_CONNECTIONS,
        (255, 235, 80),
        2,
        3,
    )


def load_helvetica_fonts():
    font_path = next((path for path in FONT_PATHS if path.is_file()), None)
    if font_path is None:
        return {size: ImageFont.load_default() for size in (13, 14, 17)}
    return {size: ImageFont.truetype(str(font_path), size) for size in (13, 14, 17)}


def numeric_sample(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def graph_sample_values(sensor_metrics):
    hr = numeric_sample(sensor_metrics.hr)
    spo2 = numeric_sample(sensor_metrics.spo2)
    confidence = numeric_sample(sensor_metrics.hr_confidence)
    finger = numeric_sample(sensor_metrics.finger)
    reliable_pulse = finger == 3 and confidence is not None and confidence >= 90 and hr is not None and hr > 0
    gsr = numeric_sample(sensor_metrics.gsr) if sensor_metrics.phase != "idle" else None
    return {
        "hr": hr if reliable_pulse else None,
        "spo2": spo2 if reliable_pulse and spo2 is not None and spo2 > 0 else None,
        "gsr": gsr,
    }


def draw_live_dashboard(frame, status, gaze_label, visual_metrics, sensor_metrics, graph_history, fonts):
    height, width = frame.shape[:2]
    header_height = 82
    graph_height = 238
    canvas = np.zeros((header_height + height + graph_height, width, 3), dtype=np.uint8)
    canvas[header_height:header_height + height, :] = frame

    graph_top = header_height + height
    panel_width = width / 3
    series = (
        ("hr", "HEART RATE", "bpm", (70, 210, 255), 40, 140),
        ("spo2", "BLOOD OXYGEN / SPO2", "%", (255, 190, 70), 85, 100),
        ("gsr", "SKIN CONDUCTANCE / GSR", "level", (90, 245, 155), None, None),
    )
    graph_labels = []

    for index, (key, title, unit, color, low, high) in enumerate(series):
        left = round(index * panel_width)
        right = round((index + 1) * panel_width)
        if index:
            cv2.line(canvas, (left, graph_top + 10), (left, graph_top + graph_height - 10), (55, 55, 55), 1)
        plot_left = left + 38
        plot_right = right - 12
        plot_top = graph_top + 42
        plot_bottom = graph_top + graph_height - 25

        values = list(graph_history[key])
        valid_values = [value for value in values if value is not None]
        if low is None and valid_values:
            low = min(valid_values) - 5
            high = max(valid_values) + 5
            if high - low < 10:
                midpoint = (high + low) / 2
                low, high = midpoint - 5, midpoint + 5
        elif low is None:
            low, high = 0, 100

        for grid_index in range(4):
            y = plot_top + round(grid_index * (plot_bottom - plot_top) / 3)
            cv2.line(canvas, (plot_left, y), (plot_right, y), (45, 45, 45), 1, cv2.LINE_AA)

        previous = None
        denominator = max(1, len(values) - 1)
        for value_index, value in enumerate(values):
            if value is None:
                previous = None
                continue
            x = plot_left + round(value_index * (plot_right - plot_left) / denominator)
            normalized = min(1.0, max(0.0, (value - low) / (high - low)))
            y = plot_bottom - round(normalized * (plot_bottom - plot_top))
            current = (x, y)
            if previous is not None:
                cv2.line(canvas, previous, current, color, 2, cv2.LINE_AA)
            previous = current
        if previous is not None:
            cv2.circle(canvas, previous, 4, color, -1, cv2.LINE_AA)

        latest = valid_values[-1] if valid_values else None
        latest_text = "--" if latest is None else f"{latest:.0f} {unit}"
        graph_labels.append((left, right, title, latest_text, high, low, plot_top, plot_bottom))

    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for left, right, title, latest_text, high, low, plot_top, plot_bottom in graph_labels:
        draw.text((left + 14, graph_top + 8), title, font=fonts[14], fill=(255, 255, 255))
        value_bounds = draw.textbbox((0, 0), latest_text, font=fonts[13])
        draw.text((right - 14 - (value_bounds[2] - value_bounds[0]), graph_top + 9), latest_text,
                  font=fonts[13], fill=(255, 255, 255))
        draw.text((left + 7, plot_top - 7), f"{high:.0f}", font=fonts[13], fill=(255, 255, 255))
        draw.text((left + 7, plot_bottom - 13), f"{low:.0f}", font=fonts[13], fill=(255, 255, 255))
    draw.text((18, 8), status, font=fonts[17], fill=(255, 255, 255))
    gaze_bounds = draw.textbbox((0, 0), gaze_label, font=fonts[14])
    draw.text((width - 18 - (gaze_bounds[2] - gaze_bounds[0]), 11), gaze_label,
              font=fonts[14], fill=(255, 255, 255))
    draw.text(
        (18, 36),
        f"Emotion: {visual_metrics.emotion} ({visual_metrics.emotion_confidence:.0f}%)  |  Blinks: {visual_metrics.blink_count}",
        font=fonts[14],
        fill=(255, 255, 255),
    )
    sensor_text = (
        f"HR {sensor_metrics.hr or '--'} bpm  |  SpO2 {sensor_metrics.spo2 or '--'}%  |  "
        f"GSR {sensor_metrics.gsr_change or '--'}%  |  {sensor_metrics.level}/{sensor_metrics.trend}"
        if sensor_metrics.connected else "Sensors offline"
    )
    draw.text((18, 58), sensor_text, font=fonts[13], fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def create_landmarker(model_path):
    options = mp_vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return mp_vision.HolisticLandmarker.create_from_options(options)


def main():
    parser = argparse.ArgumentParser(description="Preview MediaPipe Holistic landmarks.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument("--model", default="tiny.en", help="Whisper model used after the response, e.g. base.en.")
    parser.add_argument("--emotion-interval", type=int, default=EMOTION_INTERVAL, help="Analyze every N camera frames.")
    parser.add_argument("--holistic-model", type=Path, default=DEFAULT_MODEL, help="MediaPipe Holistic .task model path.")
    parser.add_argument("--sensor-port", default=None, help="QT Py serial port; auto-detected if omitted.")
    parser.add_argument("--no-sensors", action="store_true", help="Disable QT Py serial readings.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).with_name("logs"),
        help="Directory for per-session JSONL response logs.",
    )
    parser.add_argument("--silence", type=float, default=1.0, help="Seconds of silence that end an answer.")
    parser.add_argument("--max-seconds", type=float, default=60.0, help="Maximum answer length in seconds.")
    parser.add_argument("--answer-nudge-after", type=float, default=5.0, help="Seconds without speech before the first answer reminder.")
    parser.add_argument("--answer-warning-after", type=float, default=10.0, help="Seconds after the first reminder before the last-chance prompt.")
    parser.add_argument("--answer-final-after", type=float, default=15.0, help="Seconds after the last-chance prompt before ending the interview.")
    parser.add_argument(
        "--baby-sounds-dir",
        type=Path,
        default=Path(__file__).with_name("sounds"),
        help="Optional directory containing laugh, cry, and speak audio clips.",
    )
    args = parser.parse_args()
    session_id = uuid.uuid4().hex
    session_log_path = args.log_dir / f"screening-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{session_id[:8]}.jsonl"
    try:
        write_test_log(session_log_path, {"event": "session_started", "session_id": session_id})
        print(f"Session log: {session_log_path}", flush=True)
    except OSError as error:
        session_log_path = None
        print(f"Unable to start session log: {error}", flush=True)

    camera = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        raise SystemExit("Unable to open the camera. Enable camera access for Terminal or VS Code.")

    audio_queue = queue.Queue()
    dashboard = Dashboard()
    visual_metrics = VisualMetrics()
    sensor_metrics = SensorMetrics()
    graph_history = {key: deque(maxlen=GRAPH_SAMPLES) for key in ("hr", "spo2", "gsr")}
    ui_fonts = load_helvetica_fonts()
    last_graph_sample = 0.0
    calibration_music = build_calibration_music()
    calibration_music_active = False
    calibration_started_at = None
    audio_stream = None
    sensor_stop = threading.Event()
    sensor_thread = None
    if not args.no_sensors:
        sensor_port = args.sensor_port or find_sensor_port()
        if sensor_port:
            sensor_thread = threading.Thread(
                target=sensor_worker,
                args=(sensor_port, sensor_metrics, sensor_stop),
                daemon=True,
            )
            sensor_thread.start()
        else:
            sensor_metrics.error = "No QT Py found"
    emotion_queue = queue.Queue(maxsize=1)
    emotion_stop = threading.Event()
    emotion_thread = threading.Thread(
        target=emotion_worker,
        args=(emotion_queue, visual_metrics, emotion_stop),
        daemon=True,
    )
    emotion_thread.start()
    workflow_events = queue.Queue()

    def log_event(entry):
        if session_log_path is None:
            return
        try:
            write_test_log(session_log_path, {"session_id": session_id, **entry})
        except (OSError, TypeError, ValueError) as error:
            print(f"Session logging failed: {error}", flush=True)

    def audio_callback(indata, frames, callback_time, status):
        if status:
            print(f"Microphone: {status}", flush=True)
        audio_queue.put(indata[:, 0].copy())

    try:
        audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        )
        audio_stream.start()
        if not args.holistic_model.is_file():
            raise SystemExit(f"Unable to find Holistic model: {args.holistic_model}")
        with create_landmarker(args.holistic_model) as holistic:
            workflow_stage = ["waiting_for_person"]
            screening_started = False
            face_present = False
            retry_prompt_at = 0.0
            mark_stable_frames = 0
            scan_stable_frames = 0
            scan_sequence = []
            scan_index = 0
            current_scan_action = [""]
            question_queue = []
            current_question = None
            listening_started = False
            response_started = None
            prompt_finished = None
            question_started_at = None
            no_answer_level = 0
            last_speech_at = None
            recording = []
            levels = []
            speech_flags = []
            noise_samples = []
            transcription_thread = None
            frame_number = 0

            def start_instruction(text, next_action):
                workflow_stage[0] = "speaking_instruction"
                dashboard.prompt = text
                dashboard.state = "Speaking instruction"

                def speak_instruction():
                    subprocess.run(
                        ["say", text],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    workflow_events.put(("instruction_done", next_action))

                threading.Thread(target=speak_instruction, daemon=True).start()

            def stop_calibration_music():
                nonlocal calibration_music_active
                if calibration_music_active:
                    sd.stop()
                    calibration_music_active = False

            def start_calibration():
                nonlocal calibration_music_active, calibration_started_at
                if workflow_stage[0] == "calibrating":
                    return
                workflow_stage[0] = "calibrating"
                calibration_started_at = time.monotonic()
                dashboard.state = "Calibrating - please wait"
                dashboard.prompt = "Calibration - please wait"
                log_event({"event": "calibration_started", "sensor_phase": sensor_metrics.phase})
                try:
                    sd.play(calibration_music, samplerate=CALIBRATION_MUSIC_SAMPLE_RATE, loop=True)
                    calibration_music_active = True
                except Exception as error:
                    print(f"Calibration music unavailable: {error}", flush=True)

                def speak_calibration_notice():
                    subprocess.run(
                        ["say", "Calibration in progress. Please wait."],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                threading.Thread(target=speak_calibration_notice, daemon=True).start()

            def start_scan_sequence():
                nonlocal scan_sequence, scan_index, scan_stable_frames
                first_action, first_text = random.choice(SCAN_ACTIONS)
                scan_sequence = [
                    (first_action, first_text),
                    ("turn_left", TURN_LEFT_PROMPT),
                    ("turn_around", TURN_AROUND_PROMPT),
                ]
                scan_index = 0
                scan_stable_frames = 0
                action, text = scan_sequence[scan_index]
                current_scan_action[0] = action
                start_instruction(text, "scan_check")

            def start_no_answer_prompt(text, prompt_level):
                workflow_stage[0] = "speaking_no_answer_prompt"
                dashboard.state = "Speaking answer reminder"

                def speak_reminder():
                    subprocess.run(
                        ["say", text],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    while True:
                        try:
                            audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    workflow_events.put(("no_answer_prompt_done", prompt_level))

                threading.Thread(target=speak_reminder, daemon=True).start()

            def play_question_prompt(question):
                for index, part in enumerate(question["parts"]):
                    subprocess.run(
                        ["say", part],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    if index == 0 and len(question["parts"]) > 1:
                        time.sleep(question["pause_after"])

                sound_kind = question["sound"]
                if sound_kind:
                    clips = []
                    for extension in ("wav", "aiff", "mp3", "m4a"):
                        clips.extend(args.baby_sounds_dir.glob(f"{sound_kind}*.{extension}"))
                    if clips:
                        subprocess.run(
                            ["afplay", str(random.choice(clips))],
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    else:
                        print(
                            f"Missing {sound_kind} baby-sound clip in {args.baby_sounds_dir}",
                            flush=True,
                        )

            def start_next_question():
                nonlocal current_question, prompt_finished, response_started
                nonlocal last_speech_at, recording, levels, speech_flags, noise_samples
                nonlocal question_started_at, no_answer_level
                if not question_queue:
                    workflow_stage[0] = "complete"
                    dashboard.state = "Screening complete"
                    dashboard.prompt = "Screening complete"
                    return

                current_question = question_queue.pop(0)
                dashboard.prompt = current_question["text"]
                dashboard.state = "Speaking question"
                workflow_stage[0] = "speaking_question"
                prompt_finished = None
                question_started_at = None
                response_started = None
                last_speech_at = None
                no_answer_level = 0
                recording, levels, speech_flags, noise_samples = [], [], [], []

                def speak_question():
                    play_question_prompt(current_question)
                    while True:
                        try:
                            audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    workflow_events.put(("question_ready", None))

                threading.Thread(target=speak_question, daemon=True).start()

            def begin_questions():
                if workflow_stage[0] in ("asking_questions", "speaking_question", "answering", "transcribing", "complete"):
                    return
                stop_calibration_music()
                workflow_stage[0] = "asking_questions"
                question_queue[:] = [screening_prompt("Personal baseline", "What is your gender?")]
                start_next_question()

            def finish_response():
                nonlocal transcription_thread
                if not recording:
                    print("No audio captured for this answer.", flush=True)
                    return
                response = Response(
                    np.concatenate(recording),
                    list(levels),
                    list(speech_flags),
                    response_started,
                    time.monotonic(),
                )
                workflow_stage[0] = "transcribing"
                dashboard.state = "Transcribing"

                def transcribe_and_print():
                    model = WhisperModel(args.model, device="cpu", compute_type="int8")
                    transcribe(response, model, dashboard)
                    if dashboard.error:
                        print(dashboard.error, flush=True)
                        log_event({
                            "event": "transcription_error",
                            "category": current_question["category"],
                            "question": current_question["text"],
                            "error": dashboard.error,
                        })
                    else:
                        print_response_metrics(dashboard, visual_metrics, sensor_metrics)
                        log_event(response_log_entry(
                            session_id,
                            current_question,
                            dashboard,
                            visual_metrics,
                            sensor_metrics,
                        ))
                    workflow_events.put(("answer_done", None))

                transcription_thread = threading.Thread(target=transcribe_and_print, daemon=True)
                transcription_thread.start()

            def close_response():
                nonlocal response_started, prompt_finished, last_speech_at
                nonlocal listening_started, recording, levels, speech_flags
                finish_response()
                response_started = None
                prompt_finished = None
                last_speech_at = None
                listening_started = False
                recording, levels, speech_flags = [], [], []

            def handle_workflow_event(event_name, detail):
                nonlocal prompt_finished, retry_prompt_at, question_started_at
                nonlocal scan_stable_frames, no_answer_level
                if event_name == "instruction_done":
                    if detail == "scan":
                        start_instruction(MARK_PROMPT, "await_mark")
                    elif detail == "await_mark":
                        workflow_stage[0] = "waiting_for_mark"
                        retry_prompt_at = time.monotonic()
                    elif detail == "scan_check":
                        workflow_stage[0] = "waiting_for_pose"
                        retry_prompt_at = time.monotonic()
                        scan_stable_frames = 0
                    elif detail == "position":
                        start_instruction(POSITION_PROMPT, "head")
                    elif detail == "head":
                        if face_present:
                            start_instruction(HAND_PROMPT, "hand")
                        else:
                            workflow_stage[0] = "waiting_for_head"
                            retry_prompt_at = time.monotonic()
                    elif detail == "hand":
                        if sensor_metrics.connected:
                            workflow_stage[0] = "waiting_for_hand"
                            retry_prompt_at = time.monotonic()
                        else:
                            begin_questions()
                elif event_name == "question_ready":
                    prompt_finished = time.monotonic()
                    question_started_at = prompt_finished
                    no_answer_level = 0
                    workflow_stage[0] = "answering"
                    dashboard.state = "Listening - press space to finish"
                elif event_name == "no_answer_prompt_done":
                    no_answer_level = detail
                    prompt_finished = time.monotonic()
                    workflow_stage[0] = "answering"
                    dashboard.state = "Waiting for answer"
                elif event_name == "interview_ended":
                    log_event({
                        "event": "interview_ended_no_response",
                        "category": current_question["category"] if current_question else None,
                        "question": current_question["text"] if current_question else None,
                    })
                    workflow_stage[0] = "ended"
                    question_queue.clear()
                    dashboard.state = "Interview ended - no response"
                    dashboard.prompt = "Interview ended"
                elif event_name == "answer_done":
                    if dashboard.error:
                        workflow_stage[0] = "answer_error"
                    elif current_question and current_question["text"] == "What is your gender?":
                        question_queue.extend(select_followup_questions(dashboard.transcript))
                        start_next_question()
                    else:
                        start_next_question()

            while True:
                success, frame = camera.read()
                if not success:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = holistic.detect_for_video(image, frame_number * 33)
                draw_landmarks(frame, results)
                frame_number += 1
                face_present = bool(results.face_landmarks)

                if face_present:
                    face_landmarks = landmark_group(results.face_landmarks)
                    update_blink_count(face_landmarks, visual_metrics)
                    visual_metrics.looking_at_camera, visual_metrics.gaze_available = gaze_is_camera_facing(face_landmarks)
                    if screening_started and frame_number % max(1, args.emotion_interval) == 0:
                        crop = face_crop(frame, face_landmarks)
                        if crop is not None and emotion_queue.empty():
                            emotion_queue.put_nowait(crop.copy())

                if workflow_stage[0] == "waiting_for_person" and face_present:
                    screening_started = True
                    start_instruction(random.choice(GREETINGS), "scan")

                while True:
                    try:
                        event_name, detail = workflow_events.get_nowait()
                    except queue.Empty:
                        break
                    handle_workflow_event(event_name, detail)

                now = time.monotonic()
                if now - last_graph_sample >= GRAPH_SAMPLE_INTERVAL:
                    for key, value in graph_sample_values(sensor_metrics).items():
                        graph_history[key].append(value)
                    last_graph_sample = now

                if workflow_stage[0] == "waiting_for_mark":
                    mark_stable_frames = mark_stable_frames + 1 if hips_are_visible(results) else 0
                    if mark_stable_frames >= POSE_STABLE_FRAMES:
                        start_scan_sequence()
                    elif now - retry_prompt_at >= POSE_RETRY_SECONDS:
                        start_instruction(MARK_PROMPT, "await_mark")
                elif workflow_stage[0] == "waiting_for_pose":
                    action = current_scan_action[0]
                    if scan_pose_is_satisfied(action, results, face_present):
                        scan_stable_frames += 1
                    else:
                        scan_stable_frames = 0
                    if scan_stable_frames >= POSE_STABLE_FRAMES:
                        scan_index += 1
                        if scan_index < len(scan_sequence):
                            action, text = scan_sequence[scan_index]
                            current_scan_action[0] = action
                            start_instruction(text, "scan_check")
                        else:
                            start_instruction(POSITION_PROMPT, "head")
                    elif now - retry_prompt_at >= POSE_RETRY_SECONDS:
                        action, text = scan_sequence[scan_index]
                        current_scan_action[0] = action
                        scan_stable_frames = 0
                        start_instruction(text, "scan_check")
                elif workflow_stage[0] == "waiting_for_head":
                    if face_present:
                        start_instruction(HAND_PROMPT, "hand")
                    elif now - retry_prompt_at >= 4.0:
                        start_instruction(HEAD_PROMPT, "head")
                elif workflow_stage[0] == "waiting_for_hand":
                    calibration_action = sensor_calibration_action(sensor_metrics.phase, sensor_metrics.connected)
                    if calibration_action == "start":
                        start_calibration()
                    elif calibration_action == "skip":
                        begin_questions()
                    elif calibration_action == "retry" and now - retry_prompt_at >= 5.0:
                        start_instruction(HAND_PROMPT, "hand")
                elif workflow_stage[0] == "calibrating":
                    calibration_action = sensor_calibration_action(sensor_metrics.phase, sensor_metrics.connected)
                    if calibration_action == "complete":
                        stop_calibration_music()
                        log_event({
                            "event": "calibration_completed",
                            "duration_seconds": now - calibration_started_at if calibration_started_at is not None else None,
                        })
                        while True:
                            try:
                                audio_queue.get_nowait()
                            except queue.Empty:
                                break
                        begin_questions()
                    elif calibration_action in ("retry", "skip"):
                        stop_calibration_music()
                        if calibration_action == "skip":
                            log_event({"event": "calibration_aborted", "reason": "sensor_disconnected"})
                        workflow_stage[0] = "waiting_for_hand"
                        if calibration_action == "retry":
                            retry_prompt_at = now
                            start_instruction(HAND_PROMPT, "hand")
                        else:
                            begin_questions()

                while prompt_finished is not None and workflow_stage[0] == "answering":
                    try:
                        audio_block = audio_queue.get_nowait()
                    except queue.Empty:
                        break
                    level = dbfs(audio_block)
                    if response_started is None:
                        noise_samples.append(level)
                        dashboard.noise_floor = float(np.median(noise_samples[-40:]))
                        dashboard.threshold = max(-45.0, dashboard.noise_floor + 10.0)
                        is_speech = level > dashboard.threshold
                        if is_speech:
                            response_started = now
                            dashboard.response_latency = response_started - question_started_at
                            last_speech_at = now
                            listening_started = True
                            dashboard.state = "Listening - press space to finish"
                    else:
                        is_speech = level > dashboard.threshold
                        if is_speech:
                            last_speech_at = now
                    if response_started is not None:
                        recording.append(audio_block)
                        levels.append(level)
                        speech_flags.append(is_speech)

                if response_started is not None and last_speech_at is not None:
                    if now - last_speech_at >= args.silence or now - response_started >= args.max_seconds:
                        close_response()
                elif workflow_stage[0] == "answering" and response_started is None and prompt_finished is not None:
                    silent_for = now - prompt_finished
                    next_action = no_answer_action(
                        no_answer_level,
                        silent_for,
                        args.answer_nudge_after,
                        args.answer_warning_after,
                        args.answer_final_after,
                    )
                    if next_action == "reminder":
                        start_no_answer_prompt(random.choice(NO_ANSWER_PROMPTS), 1)
                    elif next_action == "last_chance":
                        start_no_answer_prompt(random.choice(LAST_CHANCE_PROMPTS), 2)
                    elif next_action == "end_interview":
                        workflow_stage[0] = "speaking_final_notice"
                        dashboard.state = "Ending interview"

                        def end_for_no_response():
                            subprocess.run(
                                ["say", END_INTERVIEW_PROMPT],
                                check=False,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            workflow_events.put(("interview_ended", None))

                        threading.Thread(target=end_for_no_response, daemon=True).start()

                if workflow_stage[0] == "waiting_for_person":
                    status = "Please enter the frame"
                elif workflow_stage[0] == "waiting_for_head":
                    status = "Waiting for face in frame"
                elif workflow_stage[0] == "waiting_for_hand":
                    status = "Waiting for sensor contact"
                elif workflow_stage[0] == "calibrating":
                    status = "Calibrating - please wait"
                elif workflow_stage[0] == "transcribing":
                    status = "Transcribing answer..."
                elif workflow_stage[0] == "complete":
                    status = "Screening complete"
                elif workflow_stage[0] == "ended":
                    status = "Interview ended - no response"
                elif workflow_stage[0] == "waiting_for_mark":
                    status = "Please stand on the marked spot"
                elif workflow_stage[0] == "waiting_for_pose":
                    status = f"Waiting for pose: {current_scan_action[0]}"
                elif workflow_stage[0] == "speaking_no_answer_prompt":
                    status = "Prompting for an answer"
                elif workflow_stage[0] == "speaking_final_notice":
                    status = "Ending interview"
                elif workflow_stage[0] == "answer_error":
                    status = "Transcription error - see terminal"
                elif workflow_stage[0] == "answering" and listening_started:
                    status = "Listening - press space to finish"
                elif workflow_stage[0] == "answering":
                    status = "Waiting for answer"
                elif workflow_stage[0] == "speaking_question":
                    status = "Speaking question"
                elif workflow_stage[0] == "speaking_instruction":
                    status = "Speaking instruction"
                else:
                    status = "Preparing screening"
                gaze_label = "Gaze: camera" if visual_metrics.looking_at_camera else "Gaze: away"
                if not visual_metrics.gaze_available:
                    gaze_label = "Gaze: unavailable"
                display_frame = draw_live_dashboard(
                    frame,
                    status,
                    gaze_label,
                    visual_metrics,
                    sensor_metrics,
                    graph_history,
                    ui_fonts,
                )
                cv2.imshow("Holistic Landmarker", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == 32 and workflow_stage[0] == "answering" and response_started is not None:
                    close_response()
    finally:
        if calibration_music_active:
            sd.stop()
        sensor_stop.set()
        emotion_stop.set()
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()