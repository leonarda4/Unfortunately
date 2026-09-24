# Facial-Emotion-Recognition-using-OpenCV-and-Deepface
This project implements real-time facial emotion detection using the `deepface` library and OpenCV. It captures video from the webcam, detects faces, and predicts the emotions associated with each face. The emotion labels are displayed on the frames in real-time.
This is probably the shortest code to implement realtime emotion monitoring.
- Give this repository a ⭐ if you liked it, since it took me time to understand and implement this
- Made with ❤️ by [Manish Tiwari](https://github.com/manish-9245)

## Dependencies

- [deepface](https://github.com/serengil/deepface): A deep learning facial analysis library that provides pre-trained models for facial emotion detection. It relies on TensorFlow for the underlying deep learning operations.
- [OpenCV](https://opencv.org/): An open-source computer vision library used for image and video processing.

## Usage
### Pulse color visualization

The separate `pulse.py` script uses a rolling Eulerian temporal bandpass to amplify
subtle color changes and micro facial movements in the detected face. The default
mode combines both effects after the history buffer warms up:

```bash
python pulse.py
```

Press `q` to quit. The defaults use a 0.7-2.0 Hz pulse band, 40x chroma
amplification, 20x motion amplification, and a five-second rolling history. Use
`--mode motion` for micro-expressions only, or `--mode color` for pulse color only.
Adjust the motion effect with `--motion-amplification`, for example:
`python pulse.py --mode motion --low 0.5 --high 3.0 --motion-amplification 15`.
The `Pulse Signal` window collapses the face ROI into a rolling red/green pulse
trace and estimates BPM from its strongest frequency. The red/green label is a
camera-color heuristic, so use the graph and BPM as the primary comparison with
your actual pulse.

### MediaPipe Holistic preview

For a plain face, pose, and hand landmark preview, use a separate environment so
MediaPipe does not conflict with the TensorFlow dependencies used by DeepFace:

```bash
python3.9 -m venv .venv-holistic
source .venv-holistic/bin/activate
pip install -r requirements-holistic.txt
python holistic.py
```

Press `q` to quit. Use `python holistic.py --camera 1` for another camera.
The Holistic environment uses the native arm64 MediaPipe Tasks package. On
older universal Python 3.9 installations, `pip check` may still report a
grpcio platform warning even though the import and runtime checks pass; Python
3.11+ arm64 removes that package metadata warning.

The Holistic preview now asks the user to lift their right hand. When the
semantic right wrist is detected above the right shoulder, it announces
“Scanning complete, come closer”, asks a question, and listens for the answer.
Press the space bar to finish the response. The Whisper transcription and
speech metrics are printed in the terminal. This requires the voice packages
listed in `requirements-holistic.txt`; the first response downloads the chosen
Whisper model. Use `python holistic.py --model base.en` for more accurate
transcription. During the come-closer phase, DeepFace displays the dominant
emotion, a green/red dot shows the gaze proxy (camera/away), and the preview
counts blinks. These are approximate signals, not clinical measurements. Use
`--emotion-interval 30` to reduce emotion-analysis frequency on a slower Mac.

### Voice interaction prototype

The `voice_interaction.py` prototype speaks a prompt with macOS text-to-speech,
listens through the microphone, ends the response after silence, transcribes it
with `faster-whisper`, and displays the tracked speech metrics in an OpenCV
window. It measures the audio signal separately from the transcript, so pauses
and response latency are retained.

Install the voice dependencies in a separate environment if desired:

```bash
python3 -m venv .venv-voice
source .venv-voice/bin/activate
pip install -r requirements-voice.txt
python voice_interaction.py
```

The first run downloads the Whisper model. Press `q` to quit, `r` to repeat the
prompt, or the space bar to end a response immediately. Adjust endpointing with
`--silence 1.5`, use `--model base.en` for more accurate transcription, or
provide a custom prompt with `--prompt "Describe your morning."`. macOS must
grant microphone access to Terminal or VS Code.

### Initial steps:
- Git clone this repository Run: `git clone https://github.com/manish-9245/Facial-Emotion-Recognition-using-OpenCV-and-Deepface.git`
- Run: `cd Facial-Emotion-Recognition-using-OpenCV-and-Deepface`
1. Install the required dependencies:
   - You can use `pip install -r requirements.txt`
   - Or you can install dependencies individually:
      - `pip install deepface`
      - `pip install tf_keras`
      - `pip install opencv-python`

2. Download the Haar cascade XML file for face detection:
   - Visit the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades) and download the `haarcascade_frontalface_default.xml` file.

3. Run the code:
   - Execute the Python script.
   - The webcam will open, and real-time facial emotion detection will start.
   - Emotion labels will be displayed on the frames around detected faces.

## Approach

1. Import the necessary libraries: `cv2` for video capture and image processing, and `deepface` for the emotion detection model.

2. Load the Haar cascade classifier XML file for face detection using `cv2.CascadeClassifier()`.

3. Start capturing video from the default webcam using `cv2.VideoCapture()`.

4. Enter a continuous loop to process each frame of the captured video.

5. Convert each frame to grayscale using `cv2.cvtColor()`.

6. Detect faces in the grayscale frame using `face_cascade.detectMultiScale()`.

7. For each detected face, extract the face ROI (Region of Interest).

8. Preprocess the face image for emotion detection using the `deepface` library's built-in preprocessing function.

9. Make predictions for the emotions using the pre-trained emotion detection model provided by the `deepface` library.

10. Retrieve the index of the predicted emotion and map it to the corresponding emotion label.

11. Draw a rectangle around the detected face and label it with the predicted emotion using `cv2.rectangle()` and `cv2.putText()`.

12. Display the resulting frame with the labeled emotion using `cv2.imshow()`.

13. If the 'q' key is pressed, exit the loop.

14. Release the video capture and close all windows using `cap.release()` and `cv2.destroyAllWindows()`.

![image](https://github.com/manish-9245/Facial-Emotion-Recognition-using-OpenCV-and-Deepface/assets/69393822/57c41270-7575-4bc7-ae7a-99d67239a5ab)



