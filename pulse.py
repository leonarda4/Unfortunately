"""Show subtle pulse and facial-expression changes with Eulerian magnification."""

import argparse
from collections import deque
from pathlib import Path
from urllib.request import urlopen

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


FACE_SIZE = (128, 128)
GRAPH_SIZE = (640, 180)
DEFAULT_FPS = 30.0
DEFAULT_MODEL = Path(__file__).with_name("models") / "holistic_landmarker.task"
DEFAULT_FACE_CASCADE = Path(__file__).with_name("haarcascade_frontalface_default.xml")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/1/holistic_landmarker.task"
)
MP_PYTHON = mp_python
MP_VISION = mp_vision
RGB_TO_YIQ = np.array(
    [[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]],
    dtype=np.float32,
)
YIQ_TO_RGB = np.linalg.inv(RGB_TO_YIQ).astype(np.float32)


def ensure_model(model_path):
    """Download the MediaPipe model once when no local model was provided."""
    model_path = Path(model_path)
    if model_path.is_file():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe Holistic Landmarker model to {model_path}...")
    try:
        with urlopen(MODEL_URL, timeout=60) as response, model_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    except Exception as error:
        model_path.unlink(missing_ok=True)
        raise SystemExit(f"Could not download the MediaPipe model: {error}") from error
    return model_path


def create_landmarker(model_path):
    options = MP_VISION.HolisticLandmarkerOptions(
        base_options=MP_PYTHON.BaseOptions(model_asset_path=str(model_path)),
        running_mode=MP_VISION.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return MP_VISION.HolisticLandmarker.create_from_options(options)


def landmark_box(face_landmarks, frame_shape, margin=0.18):
    """Convert normalized face landmarks into a padded, clipped pixel box."""
    height, width = frame_shape[:2]
    x_values = [landmark.x for landmark in face_landmarks]
    y_values = [landmark.y for landmark in face_landmarks]
    left, right = min(x_values), max(x_values)
    top, bottom = min(y_values), max(y_values)
    padding_x = (right - left) * margin
    padding_y = (bottom - top) * margin
    x = max(0, int((left - padding_x) * width))
    y = max(0, int((top - padding_y) * height))
    right = min(width, int((right + padding_x) * width))
    bottom = min(height, int((bottom + padding_y) * height))
    return x, y, max(0, right - x), max(0, bottom - y)


def detect_face_box(frame, face_cascade):
    """Return the largest detected face as a padded, clipped pixel box."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    margin_x, margin_y = int(width * 0.12), int(height * 0.12)
    left = max(0, x - margin_x)
    top = max(0, y - margin_y)
    right = min(frame.shape[1], x + width + margin_x)
    bottom = min(frame.shape[0], y + height + margin_y)
    return left, top, right - left, bottom - top


def draw_landmarks(frame, result):
    """Draw a lightweight holistic landmark overlay on the camera frame."""
    height, width = frame.shape[:2]

    def point(landmark):
        return int(landmark.x * width), int(landmark.y * height)

    if result.face_landmarks:
        for landmark in result.face_landmarks[0][::8]:
            cv2.circle(frame, point(landmark), 1, (80, 220, 120), -1)

    for landmarks, color in (
        (result.pose_landmarks, (255, 180, 0)),
        (result.left_hand_landmarks, (0, 180, 255)),
        (result.right_hand_landmarks, (255, 80, 180)),
    ):
        for landmark in landmarks:
            cv2.circle(frame, point(landmark), 2, color, -1)


def bgr_to_yiq(image):
    rgb = image[:, :, ::-1].astype(np.float32)
    return rgb @ RGB_TO_YIQ.T


def yiq_to_bgr(image):
    rgb = np.clip(image @ YIQ_TO_RGB.T, 0, 255).astype(np.uint8)
    return rgb[:, :, ::-1]


def bandpass_signal(signal, fps, low_hz, high_hz):
    """Keep only the temporal frequencies in the requested pulse band."""
    frequencies = np.fft.rfftfreq(signal.shape[0], d=1.0 / fps)
    spectrum = np.fft.rfft(signal, axis=0)
    spectrum[(frequencies < low_hz) | (frequencies > high_hz)] = 0
    return np.fft.irfft(spectrum, n=signal.shape[0], axis=0).astype(np.float32)


def pulse_channels(image):
    """Collapse a face image to spatially averaged red, green, and blue values."""
    height, width = image.shape[:2]
    margin_y, margin_x = int(height * 0.2), int(width * 0.15)
    face = image[margin_y:height - margin_y, margin_x:width - margin_x]
    channels = face.astype(np.float32).mean(axis=(0, 1))
    return channels[::-1]


def pulse_signal(history, fps, low_hz, high_hz):
    """Return the bandpassed red-minus-green pulse trace and its channel traces."""
    channels = np.stack([pulse_channels(image) for image in history], axis=0)
    normalized = channels / np.maximum(channels.mean(axis=0, keepdims=True), 1.0)
    color_signal = normalized[:, 0] - normalized[:, 1]
    filtered = bandpass_signal(color_signal, fps, low_hz, high_hz)
    return filtered, normalized


def estimate_bpm(signal, fps, low_hz, high_hz):
    """Estimate BPM from the strongest frequency in the filtered pulse band."""
    if len(signal) < 2:
        return None
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / fps)
    spectrum = np.abs(np.fft.rfft(signal - np.mean(signal)))
    valid = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(valid):
        return None
    band_indices = np.flatnonzero(valid)
    peak_index = band_indices[np.argmax(spectrum[valid])]
    return float(frequencies[peak_index] * 60.0)


def draw_pulse_graph(signal, channels, bpm, fps):
    """Render the collapsed pulse signal and label its current color tendency."""
    graph = np.full((GRAPH_SIZE[1], GRAPH_SIZE[0], 3), 24, dtype=np.uint8)
    if len(signal) < 2:
        return graph

    trace = signal / max(float(np.max(np.abs(signal))), 1e-6)
    points = np.column_stack(
        (
            np.linspace(12, GRAPH_SIZE[0] - 12, len(trace)).astype(np.int32),
            (GRAPH_SIZE[1] / 2 - trace * (GRAPH_SIZE[1] * 0.38)).astype(np.int32),
        )
    )
    red_energy = float(np.std(channels[:, 0]))
    green_energy = float(np.std(channels[:, 1]))
    color_name, color = (("RED", (40, 60, 235)) if red_energy >= green_energy else ("GREEN", (70, 210, 80)))
    cv2.line(graph, (0, GRAPH_SIZE[1] // 2), (GRAPH_SIZE[0], GRAPH_SIZE[1] // 2), (70, 70, 70), 1)
    cv2.polylines(graph, [points], False, color, 2, cv2.LINE_AA)
    bpm_text = f"{bpm:.0f} BPM" if bpm is not None else "BPM: --"
    cv2.putText(graph, f"PULSE {color_name}  |  {bpm_text}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return graph


def amplify_chroma(current_roi, history, fps, low_hz, high_hz, amplification):
    """Amplify the band-limited I/Q changes in a face ROI."""
    yiq_history = np.stack([bgr_to_yiq(image) for image in history], axis=0)
    filtered = bandpass_signal(yiq_history, fps, low_hz, high_hz)

    # Use the newest filtered frame and normalize it by recent signal strength.
    # This makes the effect visible without letting a noisy frame flash white.
    chroma_change = filtered[-1, :, :, 1:3]
    recent_strength = np.percentile(np.abs(filtered[:, :, :, 1:3]), 95)
    if recent_strength < 1e-3:
        return current_roi, 0.0

    visible_change = np.clip(chroma_change * amplification, -35.0, 35.0)
    current_yiq = bgr_to_yiq(current_roi)
    current_yiq[:, :, 1:3] += visible_change
    magnified = yiq_to_bgr(current_yiq)
    pulse_strength = float(np.mean(np.abs(visible_change)) / max(recent_strength, 1.0))
    return magnified, pulse_strength


def face_motion_band(image):
    """Return a mid-scale Laplacian detail band from a normalized face image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gaussian_1 = cv2.pyrDown(gray)
    gaussian_2 = cv2.pyrDown(gaussian_1)
    expanded = cv2.pyrUp(gaussian_2, dstsize=(gaussian_1.shape[1], gaussian_1.shape[0]))
    return gaussian_1 - expanded


def amplify_motion(current_roi, history, fps, low_hz, high_hz, amplification):
    """Amplify small, band-limited movements in the face luminance detail band."""
    motion_history = np.stack([face_motion_band(image) for image in history], axis=0)
    filtered = bandpass_signal(motion_history, fps, low_hz, high_hz)
    detail_change = filtered[-1] * amplification
    detail_change = np.clip(detail_change, -0.25, 0.25)

    current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    current_gaussian_1 = cv2.pyrDown(current_gray)
    magnified_gaussian_1 = current_gaussian_1 + detail_change
    magnified_gray = cv2.pyrUp(
        magnified_gaussian_1,
        dstsize=(current_gray.shape[1], current_gray.shape[0]),
    )

    current_yiq = bgr_to_yiq(current_roi)
    current_yiq[:, :, 0] += (magnified_gray - current_gray) * 255.0
    magnified = yiq_to_bgr(current_yiq)
    motion_strength = float(np.mean(np.abs(detail_change)))
    return magnified, motion_strength


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize subtle pulse and facial-expression changes.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Holistic Landmarker .task model path (downloaded if missing).",
    )
    parser.add_argument(
        "--mode",
        choices=("color", "motion", "both"),
        default="both",
        help="Magnification mode (default: both).",
    )
    parser.add_argument("--low", type=float, default=0.7, help="Low pulse frequency in Hz.")
    parser.add_argument("--high", type=float, default=2.0, help="High pulse frequency in Hz.")
    parser.add_argument("--amplification", type=float, default=40.0, help="Chroma amplification.")
    parser.add_argument(
        "--motion-amplification",
        type=float,
        default=20.0,
        help="Luminance motion amplification (default: 20).",
    )
    parser.add_argument("--seconds", type=float, default=5.0, help="Rolling history length.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.low <= 0 or args.high <= args.low or args.seconds <= 0:
        raise SystemExit("Use positive frequencies with --high greater than --low and --seconds > 0.")

    camera = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        raise SystemExit("Unable to open the camera. Check camera permissions and --camera.")

    fps = camera.get(cv2.CAP_PROP_FPS)
    fps = fps if 10.0 <= fps <= 120.0 else DEFAULT_FPS
    history = deque(maxlen=max(30, int(args.seconds * fps)))
    effect_strength = 0.0
    pulse_graph = np.full((GRAPH_SIZE[1], GRAPH_SIZE[0], 3), 24, dtype=np.uint8)
    bpm = None
    pulse_color = "WARMING UP"
    frame_number = 0
    face_cascade = cv2.CascadeClassifier(str(DEFAULT_FACE_CASCADE))
    if face_cascade.empty():
        raise SystemExit(f"Unable to load face detector: {DEFAULT_FACE_CASCADE}")

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_number += 1
        face_box = detect_face_box(frame, face_cascade)

        if face_box is not None:
                x, y, width, height = face_box
                roi = frame[y:y + height, x:x + width]
                if roi.size:
                    normalized = cv2.resize(roi, FACE_SIZE, interpolation=cv2.INTER_AREA)
                    history.append(normalized)
                    if len(history) >= history.maxlen:
                        history_list = list(history)
                        magnified = normalized
                        strengths = []
                        if args.mode in ("color", "both"):
                            magnified, color_strength = amplify_chroma(
                                magnified,
                                history_list,
                                fps,
                                args.low,
                                args.high,
                                args.amplification,
                            )
                            strengths.append(color_strength)
                        if args.mode in ("motion", "both"):
                            magnified, motion_strength = amplify_motion(
                                magnified,
                                history_list,
                                fps,
                                args.low,
                                args.high,
                                args.motion_amplification,
                            )
                            strengths.append(motion_strength)
                        effect_strength = max(strengths, default=0.0)
                        pulse_trace, channel_traces = pulse_signal(
                            history_list, fps, args.low, args.high
                        )
                        bpm = estimate_bpm(pulse_trace, fps, args.low, args.high)
                        pulse_color = (
                            "RED"
                            if np.std(channel_traces[:, 0]) >= np.std(channel_traces[:, 1])
                            else "GREEN"
                        )
                        pulse_graph = draw_pulse_graph(
                            pulse_trace, channel_traces, bpm, fps
                        )
                        output_roi = cv2.resize(magnified, (width, height), interpolation=cv2.INTER_LINEAR)
                        frame[y:y + height, x:x + width] = output_roi

                color = (0, 220, 255) if pulse_color == "RED" else (80, 220, 80)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

        cv2.putText(
            frame,
            f"Face magnification ({args.mode}) | warming up..."
            if len(history) < history.maxlen
            else f"Face magnification ({args.mode})",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Pulse: {pulse_color} | BPM: {bpm:.0f}" if bpm is not None else f"Pulse: {pulse_color} | BPM: --",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Pulse Visualization", frame)
        cv2.imshow("Pulse Signal", pulse_graph)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()