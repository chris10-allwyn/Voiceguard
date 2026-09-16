import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav

# ==========================================
# SETTINGS
# ==========================================

DURATION = 10
SAMPLE_RATE = 16000
TEMP_FILE = "temp_voice.wav"

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading voice encoder...")
encoder = VoiceEncoder()

# ==========================================
# FIND 10 REGISTERED RECORDINGS
# ==========================================

recordings = list(Path("recordings").glob("*"))

audio_files = [
    f for f in recordings
    if f.suffix.lower() in [".wav", ".ogg", ".mp3", ".m4a"]
]

print("\nFound", len(audio_files), "recordings.")

# ==========================================
# RECORD 10 SECONDS
# ==========================================

print("\n🎤 Get ready...")
print("Speak continuously for 10 seconds!")

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

print("\n✅ Recording complete!")

# ==========================================
# CREATE LIVE EMBEDDING
# ==========================================

live_wav = preprocess_wav(
    Path(TEMP_FILE)
)

live_embedding = encoder.embed_utterance(
    live_wav
)

live_embedding /= np.linalg.norm(live_embedding)

# ==========================================
# COMPARE WITH ALL 10 RECORDINGS
# ==========================================

print("\n==============================")
print("COMPARISON WITH ALL 10")
print("==============================")

scores = []

for file in audio_files:

    try:

        wav = preprocess_wav(file)

        embedding = encoder.embed_utterance(wav)

        embedding /= np.linalg.norm(embedding)

        similarity = np.dot(
            live_embedding,
            embedding
        )

        scores.append(similarity)

        print(
            f"{file.name:25} -> {similarity:.3f}"
        )

    except Exception as e:

        print(
            "Error processing:",
            file.name,
            e
        )

# ==========================================
# FINAL RESULT
# ==========================================

if scores:

    average = np.mean(scores)
    maximum = np.max(scores)

    print("\n==============================")
    print("FINAL RESULT")
    print("==============================")

    print(
        "Average similarity :",
        round(average, 3)
    )

    print(
        "Best similarity    :",
        round(maximum, 3)
    )

    # Temporary threshold for testing
    if maximum >= 0.65:

        print("\n✅ REGISTERED USER")

    else:

        print("\n❌ UNKNOWN SPEAKER")

    print("==============================")