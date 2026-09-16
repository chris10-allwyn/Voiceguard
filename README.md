# VoiceGuard

VoiceGuard is a Python project that explores detecting AI-generated speech and checking whether a voice matches a registered speaker. It combines audio analysis with speaker verification to help flag potentially suspicious audio.

> **Note:** VoiceGuard is a prototype, not a security guarantee. Model predictions can be wrong. Do not use it as the sole basis for important decisions.

## Features

* Analyzes audio for signs of AI-generated speech.
* Compares audio with a registered speaker profile.
* Supports microphone-based audio testing, if configured.
* Displays or reports detection and verification results.

## Requirements

* Python 3.10 or a compatible version
* Git
* A working microphone for microphone-based features
* The Python packages listed in `requirements.txt` (if included)

## Setup

Clone the repository:

```bash
git clone https://github.com/chris10-allwyn/Voiceguard.git
cd Voiceguard
```

Create and activate a virtual environment:

**Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If the project does not yet have a `requirements.txt`, install the dependencies listed in the project’s setup instructions.

## Usage

Run the script for the feature you want to test. For example:

```bash
python voiceguard_live.py
```

Check the project files for the correct script name and any required setup steps. Some features may require trained model files or a registered voice profile.

## Project Structure

```text
VoiceGuard/
├── models/             # Model files and related data, if provided
├── ai_detector.py      # AI-generated speech detection
├── speaker_verify.py   # Speaker verification
├── voiceguard_live.py  # Live microphone workflow
└── README.md
```

Update this tree so it matches the files in your repository.

## Privacy

Audio recordings and voice profiles are sensitive personal data. Keep them private, obtain permission before recording or analyzing someone’s voice, and do not upload private recordings or voice-profile files to GitHub.

## Limitations

* Results depend on the models, audio quality, and recording conditions.
* The detector may produce false positives or false negatives.
* Microphone testing does not automatically mean the project can analyze phone calls or other apps’ audio.
* This project is for learning and experimentation.

## Future Improvements

* Improve detection accuracy and evaluate it on diverse audio samples.
* Add clearer result reporting and error handling.
* Document model training and evaluation.
* Add automated tests.

## License

No license has been specified yet. Add a license before allowing others to reuse or distribute the project.
