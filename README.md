# Unfortunately

A camera-based interaction prototype for voice and nonverbal signals.

The MediaPipe Holistic interaction greets a person when they enter the camera
frame, gives scan and positioning prompts, checks face framing and sensor
contact, then asks a randomized sequence of screening questions. Each answer is
transcribed with Whisper and its speech metrics are printed in the terminal.

During the interaction it also shows:

- Face, pose, and hand landmarks
- A coarse gaze-to-camera proxy: green means camera-facing, red means away
- A blink counter based on eye aspect ratio
- Background DeepFace emotion analysis during the come-closer phase

These visual signals are approximate measurements and are not clinical
assessments. This prototype asks sensitive employment, family, and health
questions and captures voice, face, and sensor signals. Do not use its output
to make employment decisions; obtain informed consent and review applicable
privacy and employment requirements before any use.

## Setup

This project targets Apple Silicon macOS. Use the native Python environment:

```bash
python3 -m pip install --user uv
export PATH="$HOME/Library/Python/3.9/bin:$PATH"
uv python install 3.11
uv venv --python 3.11 .venv-holistic
source .venv-holistic/bin/activate
uv pip install -r requirements-holistic.txt
python holistic.py
```

macOS must grant camera and microphone access to Terminal or VS Code. The first
response downloads the selected Whisper model. Use `--model base.en` for more
accurate transcription, or `--emotion-interval 30` on slower machines.
If a QT Py sensor board is connected, its latest heart rate, SpO2, GSR change,
level, trend, and spike count appear during the come-closer phase and are
included in the terminal summary. Use `--sensor-port /dev/cu.usbmodemXXXX` to
select a port, or `--no-sensors` to disable serial readings.
After sensor contact starts its baseline, the app says “Calibration in
progress. Please wait.” and plays a synthesized elevator-style instrumental
loop until the sensor enters its measuring phase. If the sensor disconnects,
the music stops and the screening continues without sensor calibration.
Answers end after one second of silence by default; use `--silence` and
`--max-seconds` to adjust answer capture. The women-specific audio prompt looks
for optional clips named `laugh*.wav`, `cry*.wav`, or `speak*.wav` in `sounds/`;
set `--baby-sounds-dir` to use another directory. If no matching clip exists,
the spoken prompt still runs and the missing clip is reported in the terminal.
The pose scan waits until both hips are visible, checks each lift/turn before
continuing, and repeats an uncompleted pose prompt every four seconds. If an
answer never starts, the app gives a reminder after five seconds, a last-chance
prompt after another ten seconds, and ends the interview after another fifteen
seconds. Adjust those waits with `--answer-nudge-after`,
`--answer-warning-after`, and `--answer-final-after`.
Each run writes a local JSONL log under `logs/` by default; use `--log-dir` to
choose another directory. It records each question, transcript, response and
signal metrics, and calibration/interview events. Raw microphone audio is not
logged. Treat these files as sensitive personal data and only retain them with
appropriate consent and safeguards.

## Controls

- Enter the camera frame to begin the interaction.
- Answer each spoken question; one second of silence advances to the next.
- Follow each lift/turn prompt; the next scan prompt waits for the pose check.
- Press Space to finish the current answer immediately.
- Session records are written to `logs/`.
- Press `q` to quit.

## Credits

The original facial-emotion prototype and initial project materials came from
[Manish Tiwari](https://github.com/manish-9245). The current repository is a
clean rewrite focused on the Holistic interaction and is being extended with
the separate `sensor-hardware` branch next.
