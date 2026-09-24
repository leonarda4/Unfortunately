# Unfortunately

A camera-based interaction prototype for voice and nonverbal signals.

The current starting point is the MediaPipe Holistic interaction. It asks the
user to lift their right hand, says "Scanning complete, come closer", asks a
question, listens for a spoken response, and prints speech metrics and a
Whisper transcription in the terminal.

During the interaction it also shows:

- Face, pose, and hand landmarks
- A coarse gaze-to-camera proxy: green means camera-facing, red means away
- A blink counter based on eye aspect ratio
- Background DeepFace emotion analysis during the come-closer phase

These visual signals are approximate measurements and are not clinical
assessments.

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

## Controls

- Raise the user's right hand to begin the interaction.
- Press Space to finish the spoken response.
- Press `q` to quit.

## Credits

The original facial-emotion prototype and initial project materials came from
[Manish Tiwari](https://github.com/manish-9245). The current repository is a
clean rewrite focused on the Holistic interaction and is being extended with
the separate `sensor-hardware` branch next.
