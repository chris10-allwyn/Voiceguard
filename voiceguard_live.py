import sounddevice as sd
import soundfile as sf
import numpy as np

from pathlib import Path

from ai_detector import detect_ai_voice


# =========================
# SETTINGS
# =========================

DURATION = 5
SAMPLE_RATE = 16000

TEMP_FILE = "live_audio.wav"

AI_THRESHOLD = 50


# =========================
# START
# =========================

print("======================================")
print("        🛡️ VOICEGUARD")
print("======================================")

print("\nAI Voice Detection System")
print("Press CTRL + C to stop.\n")


try:

    while True:

        print("--------------------------------------")
        print("🎤 Listening for 5 seconds...")
        print("Speak normally...")

        # Record microphone

        audio = sd.rec(
            int(
                DURATION *
                SAMPLE_RATE
            ),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        # Save audio

        sf.write(
            TEMP_FILE,
            audio,
            SAMPLE_RATE
        )

        print("✅ Recording complete.")

        # =========================
        # AI DETECTION
        # =========================

        result, probability = detect_ai_voice(
            TEMP_FILE
        )

        print("\n======================================")
        print("          VOICE ANALYSIS")
        print("======================================")

        print(
            "AI Probability:",
            round(probability, 2),
            "%"
        )

        print(
            "Detection:",
            result
        )

        # =========================
        # RISK
        # =========================

        if probability >= 85:

            print("\n🚨 HIGH RISK")
            print("AI-generated voice strongly detected!")

        elif probability >= 60:

            print("\n⚠️ MEDIUM RISK")
            print("Voice requires further verification.")

        else:

            print("\n🟢 LOW RISK")
            print("Voice appears to be real.")

        print("======================================\n")


except KeyboardInterrupt:

    print("\n\nVoiceGuard stopped.")