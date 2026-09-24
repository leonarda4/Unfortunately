"""Live MediaPipe Holistic face, pose, and hand landmark preview."""

import argparse
import subprocess

import cv2
import mediapipe as mp


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


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

    wrist = results.right_hand_landmarks[0].landmark[mp_holistic.HandLandmark.WRIST]
    shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
    return wrist.y < shoulder.y - 0.05


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
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        raise SystemExit("Unable to open the camera. Enable camera access for Terminal or VS Code.")

    try:
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
                        speak("Scanning complete, come closer")
                        completion_announced = True

                if completion_announced:
                    status = "Scanning complete - come closer"
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
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()