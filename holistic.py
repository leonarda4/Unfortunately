"""Live MediaPipe Holistic face, pose, and hand landmark preview."""

import argparse
import queue
import subprocess
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from voice_interaction import BLOCK_SIZE, SAMPLE_RATE, Dashboard, Response, compute_metrics, dbfs, transcribe


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
QUESTION = "What is something that made you feel proud recently?"


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

    wrist = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
    shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
    return wrist.y < shoulder.y - 0.05


def print_response_metrics(dashboard):
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
    print("----------------------\n", flush=True)


def draw_landmarks(frame, results):
    mp_drawing.draw_landmarks(
        frame,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
    )
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
    )
    mp_drawing.draw_landmarks(
        frame,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style(),
    )
    mp_drawing.draw_landmarks(
        frame,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style(),
    )


def main():
    parser = argparse.ArgumentParser(description="Preview MediaPipe Holistic landmarks.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument("--model", default="tiny.en", help="Whisper model used after the response, e.g. base.en.")
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        raise SystemExit("Unable to open the camera. Enable camera access for Terminal or VS Code.")

    audio_queue = queue.Queue()
    dashboard = Dashboard(prompt=QUESTION)
    audio_stream = None

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
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
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
                        print_response_metrics(dashboard)

                transcription_thread = threading.Thread(target=transcribe_and_print, daemon=True)
                transcription_thread.start()

            while True:
                success, frame = camera.read()
                if not success:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = holistic.process(rgb_frame)
                rgb_frame.flags.writeable = True
                draw_landmarks(frame, results)

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
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()