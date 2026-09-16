import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav

# ==========================================
# SETTINGS
# ==========================================

DURATION = 2          # Record 2 seconds
SAMPLE_RATE = 16000
TEMP_FILE = "temp_voice.wav"

# ==========================================
# LOAD VOICE MODEL
# ==========================================

print("Loading voice model...")
encoder = VoiceEncoder()

# ==========================================
# LOAD 10-RECORDING VOICE PROFILE
# ==========================================

profile_path = Path("models/user_profile.npy")

if not profile_path.exists():
    print("ERROR: user_profile.npy not found!")
    print("Run create_profile.py first.")
    exit()

user_profile = np.load(profile_path)

print("Voice profile loaded.")
print("Microphone ready!\n")


# ==========================================
# RECORD FROM MICROPHONE
# ==========================================

def record_audio():

    print("🎤 Speak now...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    sf.write(
        TEMP_FILE,
        audio,
        SAMPLE_RATE
    )

    print("Recording complete.")


# ==========================================
# COMPARE WITH REGISTERED VOICE
# ==========================================

def compare_audio():

    wav = preprocess_wav(Path(TEMP_FILE))

    live_embedding = encoder.embed_utterance(wav)

    # Normalize
    live_embedding = (
        live_embedding /
        np.linalg.norm(live_embedding)
    )

    # Cosine similarity
    similarity = np.dot(
        user_profile,
        live_embedding
    )

    return float(similarity)


# ==========================================
# CONTINUOUS MICROPHONE VERIFICATION
# ==========================================

while True:

    input("Press ENTER to record...")

    record_audio()

    similarity = compare_audio()

    print("\n------------------------------")
    print("VOICE VERIFICATION")
    print("------------------------------")
    print("Similarity:", round(similarity, 3))

    if similarity >= 0.75:
        print("✅ MATCH — Registered User")
    else:
        print("❌ NOT MATCH — Unknown Voice")

    print("------------------------------\n")

    choice = input(
        "Press ENTER to continue or type q to quit: "
    )

    if choice.lower() == "q":
        break

print("\nVoiceGuard stopped.")