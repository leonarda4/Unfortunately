"""Live MediaPipe Holistic face, pose, and hand landmark preview."""

import argparse
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from voice_interaction import BLOCK_SIZE, SAMPLE_RATE, Dashboard, Response, compute_metrics, dbfs, transcribe


QUESTION = "What is something that made you feel proud recently?"
DEFAULT_MODEL = Path(__file__).with_name("models") / "holistic_landmarker.task"
EMOTION_INTERVAL = 15
BLINK_THRESHOLD = 0.21
BLINK_MIN_FRAMES = 2


@dataclass
class VisualMetrics:
    emotion: str = "warming up"
    emotion_confidence: float = 0.0
    emotion_error: str = ""
    looking_at_camera: bool = False
    gaze_available: bool = False
    blink_count: int = 0
    eyes_closed_frames: int = 0


def speak(message):
    """Speak without blocking the camera preview on macOS."""
    subprocess.Popen(
        ["say", message],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def right_hand_is_raised(results):
    """Return true when the user's semantic right wrist is above the shoulder."""
    if not results.right_hand_landmarks or not results.pose_landmarks:
        return False

    if not results.right_hand_landmarks[0] or not results.pose_landmarks[0]:
        return False
    wrist = results.right_hand_landmarks[0][0]
    shoulder = results.pose_landmarks[0][12]
    return wrist.y < shoulder.y - 0.05


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


def print_response_metrics(dashboard, visual_metrics):
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
    print("----------------------\n", flush=True)


def draw_landmarks(frame, results):
    height, width = frame.shape[:2]

    def point(landmark):
        return int(landmark.x * width), int(landmark.y * height)

    if results.face_landmarks:
        for landmark in results.face_landmarks[0][::8]:
            cv2.circle(frame, point(landmark), 1, (80, 220, 120), -1)
    for landmarks, color in (
        (results.pose_landmarks, (255, 180, 0)),
        (results.left_hand_landmarks, (0, 180, 255)),
        (results.right_hand_landmarks, (255, 80, 180)),
    ):
        if landmarks:
            for landmark in landmarks[0]:
                cv2.circle(frame, point(landmark), 2, color, -1)


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
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        raise SystemExit("Unable to open the camera. Enable camera access for Terminal or VS Code.")

    audio_queue = queue.Queue()
    dashboard = Dashboard(prompt=QUESTION)
    visual_metrics = VisualMetrics()
    audio_stream = None
    emotion_queue = queue.Queue(maxsize=1)
    emotion_stop = threading.Event()
    emotion_thread = threading.Thread(
        target=emotion_worker,
        args=(emotion_queue, visual_metrics, emotion_stop),
        daemon=True,
    )
    emotion_thread.start()

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
            speak("Please lift your right hand")
            hand_detected = False
            completion_announced = False
            listening_started = False
            response_started = None
            prompt_finished = None
            recording = []
            levels = []
            speech_flags = []
            noise_samples = []
            transcription_thread = None
            frame_number = 0

            def ask_question():
                nonlocal prompt_finished
                subprocess.run(
                    ["say", "Scanning complete, come closer."],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    ["say", QUESTION],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                while not audio_queue.empty():
                    audio_queue.get_nowait()
                prompt_finished = time.monotonic()

            def finish_response():
                nonlocal transcription_thread
                if not recording:
                    print("No audio captured. Speak your answer, then press space.", flush=True)
                    return
                response = Response(
                    np.concatenate(recording),
                    list(levels),
                    list(speech_flags),
                    response_started,
                    time.monotonic(),
                )
                dashboard.state = "Transcribing"

                def transcribe_and_print():
                    model = WhisperModel(args.model, device="cpu", compute_type="int8")
                    transcribe(response, model, dashboard)
                    if dashboard.error:
                        print(dashboard.error, flush=True)
                    else:
                        print_response_metrics(dashboard, visual_metrics)

                transcription_thread = threading.Thread(target=transcribe_and_print, daemon=True)
                transcription_thread.start()

            while True:
                success, frame = camera.read()
                if not success:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = holistic.detect_for_video(image, frame_number * 33)
                draw_landmarks(frame, results)
                frame_number += 1

                if results.face_landmarks:
                    face_landmarks = results.face_landmarks[0]
                    update_blink_count(face_landmarks, visual_metrics)
                    visual_metrics.looking_at_camera, visual_metrics.gaze_available = gaze_is_camera_facing(face_landmarks)
                    if completion_announced and frame_number % max(1, args.emotion_interval) == 0:
                        crop = face_crop(frame, face_landmarks)
                        if crop is not None and emotion_queue.empty():
                            emotion_queue.put_nowait(crop.copy())

                if not hand_detected and right_hand_is_raised(results):
                    hand_detected = True
                    if not completion_announced:
                        completion_announced = True
                        dashboard.state = "Asking question"
                        threading.Thread(target=ask_question, daemon=True).start()

                now = time.monotonic()
                while prompt_finished is not None:
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
                            dashboard.response_latency = response_started - prompt_finished
                            listening_started = True
                            dashboard.state = "Listening - press space to finish"
                    else:
                        is_speech = level > dashboard.threshold
                    if response_started is not None:
                        recording.append(audio_block)
                        levels.append(level)
                        speech_flags.append(is_speech)

                if completion_announced:
                    if dashboard.state == "Asking question":
                        status = "Come closer - answering question"
                    elif dashboard.state == "Transcribing":
                        status = "Transcribing response..."
                    elif dashboard.state.startswith("Complete"):
                        status = "Response complete - see terminal"
                    elif listening_started:
                        status = "Listening - press space to finish"
                    else:
                        status = "Come closer - waiting for answer"
                    status_color = (80, 240, 120)
                else:
                    status = "Please lift your right hand"
                    status_color = (0, 220, 255)

                cv2.putText(
                    frame,
                    f"{status} | press q to quit",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2,
                )
                gaze_color = (0, 220, 0) if visual_metrics.looking_at_camera else (0, 0, 255)
                gaze_label = "Gaze: camera" if visual_metrics.looking_at_camera else "Gaze: away"
                if not visual_metrics.gaze_available:
                    gaze_label = "Gaze: unavailable"
                    gaze_color = (150, 150, 150)
                cv2.circle(frame, (frame.shape[1] - 45, 45), 12, gaze_color, -1)
                cv2.putText(frame, gaze_label, (frame.shape[1] - 230, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, gaze_color, 2)
                cv2.putText(frame, f"Emotion: {visual_metrics.emotion} ({visual_metrics.emotion_confidence:.0f}%)", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 100), 2)
                cv2.putText(frame, f"Blinks: {visual_metrics.blink_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 100), 2)
                cv2.imshow("Holistic Landmarker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == 32 and response_started is not None:
                    finish_response()
                    response_started = None
                    prompt_finished = None
                    listening_started = False
                    recording, levels, speech_flags = [], [], []
    finally:
        emotion_stop.set()
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()