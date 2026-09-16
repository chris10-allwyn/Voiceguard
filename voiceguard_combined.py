
"""
VoiceGuard Combined Prototype
-----------------------------
Runs the existing AI voice detector and speaker verifier on the
same 5-second microphone chunk.

Expected files:
    VoiceGuard/
        voiceguard_combined.py
        ai_detector.py
        speaker_verify.py
        models/
            ai_voice_detector.pth
            user_profile.npy

Run from this folder:
    python voiceguard_combined.py

Press Ctrl+C to stop.
"""

from collections import deque
from datetime import datetime
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np
import sounddevice as sd
import soundfile as sf


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

SAMPLE_RATE = 16000
CHUNK_SECONDS = 5
CHANNELS = 1

# Analyze the most recent 3 chunks.
ROLLING_WINDOW = 3

# AI detector returns a percentage from 0 to 100.
AI_HIGH_THRESHOLD = 85.0
AI_REVIEW_THRESHOLD = 60.0

# Speaker similarity threshold.
# This is a starting point, NOT a universal threshold.
# Calibrate using your own genuine and non-matching recordings.
SPEAKER_MATCH_THRESHOLD = 0.76

# Avoid treating almost-silent chunks as reliable predictions.
MIN_RMS_LEVEL = 0.003

# Store temporary chunk audio beside this script.
TEMP_DIR = BASE_DIR / "temp_chunks"
TEMP_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# LOAD THE EXISTING MODELS
# ------------------------------------------------------------

# The existing model files use paths relative to the project
# folder, so set the working directory before importing them.
import os
os.chdir(BASE_DIR)

try:
    from ai_detector import detect_ai_voice
    from speaker_verify import compare_voice
except Exception as exc:
    raise RuntimeError(
        "Could not load the existing models or their dependencies.\n"
        "Check that models/ai_voice_detector.pth and "
        "models/user_profile.npy exist, and install the project's "
        "required packages.\n"
        f"Original error: {exc}"
    ) from exc


# ------------------------------------------------------------
# AUDIO UTILITIES
# ------------------------------------------------------------

def calculate_rms(audio):
    """Return the root-mean-square audio level."""
    audio = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(audio))))


def record_chunk():
    """Record exactly one 5-second microphone chunk."""
    print(f"\n🎤 Listening for {CHUNK_SECONDS} seconds...")

    audio = sd.rec(
        int(CHUNK_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )

    sd.wait()

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if len(audio) == 0:
        raise RuntimeError("Microphone returned an empty recording.")

    return audio


def save_chunk(audio):
    """Save audio as a temporary WAV file and return its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = TEMP_DIR / f"chunk_{timestamp}.wav"

    sf.write(
        str(path),
        audio,
        SAMPLE_RATE,
        subtype="PCM_16",
    )

    return path


# ------------------------------------------------------------
# RUN BOTH EXISTING MODELS
# ------------------------------------------------------------

def analyze_chunk(audio_path):
    """
    Run the AI detector and speaker verifier on the same WAV file.

    Returns a dictionary containing both model outputs.
    """
    # Existing AI detector:
    # returns ("REAL" or "AI-GENERATED", AI percentage)
    ai_label, ai_probability = detect_ai_voice(str(audio_path))
    ai_probability = float(ai_probability)

    # Existing speaker verifier:
    # returns cosine similarity against models/user_profile.npy
    speaker_similarity = float(compare_voice(str(audio_path)))

    if not np.isfinite(ai_probability):
        raise ValueError("AI detector returned an invalid probability.")

    if not np.isfinite(speaker_similarity):
        raise ValueError("Speaker verifier returned an invalid score.")

    return {
        "ai_label": str(ai_label),
        "ai_probability": ai_probability,
        "speaker_similarity": speaker_similarity,
    }


# ------------------------------------------------------------
# DECISION ENGINE
# ------------------------------------------------------------

def decide_risk(ai_probability, speaker_similarity):
    """
    Interpret the two model outputs.

    This is a prototype rule-based decision engine, not a
    validated fraud detector.
    """
    ai_high = ai_probability >= AI_HIGH_THRESHOLD
    ai_review = ai_probability >= AI_REVIEW_THRESHOLD

    speaker_match = (
        speaker_similarity >= SPEAKER_MATCH_THRESHOLD
    )

    if ai_high:
        return {
            "status": "HIGH RISK",
            "reason": "AI detector returned a high probability.",
            "alert": True,
        }

    if ai_review:
        return {
            "status": "NEEDS VERIFICATION",
            "reason": "AI probability is in the review range.",
            "alert": False,
        }

    if not speaker_match:
        return {
            "status": "NEEDS VERIFICATION",
            "reason": "Voice did not meet the enrolled-speaker threshold.",
            "alert": False,
        }

    return {
        "status": "PROVISIONAL SAFE",
        "reason": "Low AI probability and enrolled-speaker match.",
        "alert": False,
    }


# ------------------------------------------------------------
# ROLLING ANALYSIS
# ------------------------------------------------------------

def analyze_rolling_window(history):
    """
    Summarize recent chunk results.

    HIGH RISK:
        Any recent chunk is above the high AI threshold.

    NEEDS VERIFICATION:
        At least 2 recent chunks are in the AI review range,
        or the latest chunk fails speaker verification.

    PROVISIONAL SAFE:
        All chunks in the rolling window have low AI probability
        and meet the speaker similarity threshold.
    """
    if not history:
        return {
            "status": "WAITING",
            "reason": "No analyzed chunks yet.",
            "alert": False,
        }

    recent = list(history)

    high_count = sum(
        item["ai_probability"] >= AI_HIGH_THRESHOLD
        for item in recent
    )

    review_count = sum(
        item["ai_probability"] >= AI_REVIEW_THRESHOLD
        for item in recent
    )

    latest = recent[-1]

    if high_count > 0:
        return {
            "status": "HIGH RISK",
            "reason": (
                f"{high_count} recent chunk(s) had high AI probability."
            ),
            "alert": True,
        }

    if review_count >= 2:
        return {
            "status": "NEEDS VERIFICATION",
            "reason": (
                f"{review_count} recent chunk(s) require AI review."
            ),
            "alert": False,
        }

    if (
        latest["speaker_similarity"]
        < SPEAKER_MATCH_THRESHOLD
    ):
        return {
            "status": "NEEDS VERIFICATION",
            "reason": (
                "Latest chunk did not meet speaker similarity threshold."
            ),
            "alert": False,
        }

    if len(recent) < ROLLING_WINDOW:
        return {
            "status": "COLLECTING",
            "reason": (
                f"Collecting rolling window: "
                f"{len(recent)}/{ROLLING_WINDOW} chunks."
            ),
            "alert": False,
        }

    all_low_ai = all(
        item["ai_probability"] < AI_REVIEW_THRESHOLD
        for item in recent
    )

    all_speaker_match = all(
        item["speaker_similarity"] >= SPEAKER_MATCH_THRESHOLD
        for item in recent
    )

    if all_low_ai and all_speaker_match:
        return {
            "status": "PROVISIONAL SAFE",
            "reason": (
                "All recent chunks had low AI probability "
                "and met the speaker threshold."
            ),
            "alert": False,
        }

    return {
        "status": "NEEDS VERIFICATION",
        "reason": "Recent results are not consistently low risk.",
        "alert": False,
    }


# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

def print_chunk_result(number, result, decision):
    print("\n" + "=" * 55)
    print(f"CHUNK #{number}")
    print("=" * 55)

    print(
        f"AI detector label:       {result['ai_label']}"
    )
    print(
        f"AI probability:          "
        f"{result['ai_probability']:.2f}%"
    )
    print(
        f"Speaker similarity:     "
        f"{result['speaker_similarity']:.4f}"
    )
    print(
        f"Speaker threshold:      "
        f"{SPEAKER_MATCH_THRESHOLD:.2f}"
    )

    print("-" * 55)
    print(f"Chunk decision: {decision['status']}")
    print(f"Reason: {decision['reason']}")
    print("=" * 55)


def print_rolling_result(decision, history):
    print("\n" + "-" * 55)
    print("ROLLING ANALYSIS")
    print("-" * 55)

    print(
        f"Window: {len(history)}/{ROLLING_WINDOW} chunks"
    )
    print(f"Current status: {decision['status']}")
    print(f"Reason: {decision['reason']}")

    if decision["alert"]:
        print("\n🚨 ALERT: HIGH RISK DETECTED")
    elif decision["status"] == "PROVISIONAL SAFE":
        print("\n🟢 PROVISIONAL SAFE — continue monitoring")
    elif decision["status"] == "COLLECTING":
        print("\n⏳ Collecting more audio...")
    else:
        print("\n⚠️ Verification required")

    print("-" * 55)


# ------------------------------------------------------------
# MAIN LIVE LOOP
# ------------------------------------------------------------

def main():
    history = deque(maxlen=ROLLING_WINDOW)
    chunk_number = 0

    print("\n" + "=" * 55)
    print("             VOICEGUARD COMBINED")
    print("=" * 55)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Chunk duration: {CHUNK_SECONDS} seconds")
    print(f"Rolling window: {ROLLING_WINDOW} chunks")
    print(f"AI high threshold: {AI_HIGH_THRESHOLD:.1f}%")
    print(f"AI review threshold: {AI_REVIEW_THRESHOLD:.1f}%")
    print(
        f"Speaker threshold: {SPEAKER_MATCH_THRESHOLD:.2f}"
    )
    print("\nPress Ctrl+C to stop.")
    print(
        "\nThis prototype analyzes microphone audio only. "
        "It does not capture telephone calls."
    )

    try:
        while True:
            chunk_number += 1
            audio_path = None

            try:
                audio = record_chunk()

                rms = calculate_rms(audio)

                if rms < MIN_RMS_LEVEL:
                    print(
                        "\n⚠️ Audio is too quiet for reliable analysis."
                    )
                    print(
                        "Please check the microphone and speak "
                        "clearly. This chunk is not classified."
                    )
                    continue

                audio_path = save_chunk(audio)

                print("Running AI detector and speaker verification...")

                # Both models receive the same saved audio chunk.
                result = analyze_chunk(audio_path)

                # Chunk-level decision
                chunk_decision = decide_risk(
                    result["ai_probability"],
                    result["speaker_similarity"],
                )

                print_chunk_result(
                    chunk_number,
                    result,
                    chunk_decision,
                )

                # Add result to rolling history
                history.append(result)

                # Rolling decision
                rolling_decision = analyze_rolling_window(history)

                print_rolling_result(
                    rolling_decision,
                    history,
                )

            except KeyboardInterrupt:
                raise

            except Exception as exc:
                print("\n❌ Could not analyze this chunk.")
                print(f"Error: {exc}")
                print(
                    "No safe classification was made. "
                    "Check the model files and dependencies."
                )

                # Do not add failed chunks to rolling history.

            finally:
                # Remove the temporary WAV after both models finish.
                if audio_path is not None:
                    try:
                        audio_path.unlink(missing_ok=True)
                    except OSError:
                        pass

    except KeyboardInterrupt:
        print("\n\nVoiceGuard stopped by user.")

    finally:
        print("Microphone monitoring ended.")


if __name__ == "__main__":
    main()