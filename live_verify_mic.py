import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav


# ==========================================
# SETTINGS
# ==========================================

DURATION = 5
SAMPLE_RATE = 16000
TEMP_FILE = "live_chunk.wav"

# Thresholds based on calibration results
BEST_SCORE_THRESHOLD = 0.88
TOP3_AVERAGE_THRESHOLD = 0.86
OVERALL_AVERAGE_THRESHOLD = 0.76


# ==========================================
# LOAD VOICE MODEL
# ==========================================

print("Loading voice model...")

encoder = VoiceEncoder()

print("Voice model loaded!")


# ==========================================
# LOAD ENROLLED VOICE RECORDINGS
# ==========================================

recordings_folder = Path("mic_recordings")

if not recordings_folder.exists():

    print("\nERROR: mic_recordings folder not found!")
    print("Run record_enrollment.py first.")

    exit()


audio_extensions = [
    ".wav",
    ".ogg",
    ".mp3",
    ".m4a"
]


recordings = sorted([
    file
    for file in recordings_folder.iterdir()
    if file.suffix.lower() in audio_extensions
])


if len(recordings) == 0:

    print("ERROR: No enrollment recordings found!")

    exit()


print("\nFound", len(recordings), "registered recordings.")


# ==========================================
# CREATE REGISTERED VOICE EMBEDDINGS
# ==========================================

print("\nCreating registered voice embeddings...\n")

registered_embeddings = []
registered_names = []


for file in recordings:

    try:

        print("Processing:", file.name)

        wav = preprocess_wav(file)

        embedding = encoder.embed_utterance(wav)

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        registered_embeddings.append(embedding)

        registered_names.append(file.name)

    except Exception as e:

        print("\nError processing:", file.name)
        print(e)


if len(registered_embeddings) == 0:

    print("ERROR: Could not process enrollment recordings!")

    exit()


print("\n======================================")
print("        VOICEGUARD READY")
print("======================================")

print("Registered samples:", len(registered_embeddings))
print("Chunk duration:", DURATION, "seconds")

print("======================================\n")


# ==========================================
# CONTINUOUS LIVE VERIFICATION
# ==========================================

try:

    while True:

        print("\n🎤 Listening for", DURATION, "seconds...")
        print("Speak normally...")


        # ==========================================
        # RECORD MICROPHONE AUDIO
        # ==========================================

        audio = sd.rec(

            int(DURATION * SAMPLE_RATE),

            samplerate=SAMPLE_RATE,

            channels=1,

            dtype="float32"

        )

        sd.wait()


        # Save current audio chunk
        sf.write(

            TEMP_FILE,

            audio,

            SAMPLE_RATE

        )


        print("Recording complete.")


        # ==========================================
        # PROCESS LIVE AUDIO
        # ==========================================

        try:

            wav = preprocess_wav(
                Path(TEMP_FILE)
            )

            live_embedding = encoder.embed_utterance(
                wav
            )


            # Normalize embedding
            live_embedding = (

                live_embedding /

                np.linalg.norm(live_embedding)

            )


        except Exception as e:

            print("Error processing live audio:", e)

            continue


        # ==========================================
        # COMPARE WITH ALL REGISTERED SAMPLES
        # ==========================================

        results = []


        for i, registered_embedding in enumerate(
            registered_embeddings
        ):

            similarity = float(

                np.dot(

                    live_embedding,

                    registered_embedding

                )

            )


            results.append({

                "name": registered_names[i],

                "score": similarity

            })


        # Sort highest score first
        results.sort(

            key=lambda x: x["score"],

            reverse=True

        )


        # ==========================================
        # CALCULATE SCORES
        # ==========================================

        top_3 = results[:3]


        best_score = top_3[0]["score"]


        top3_average = np.mean([

            item["score"]

            for item in top_3

        ])


        overall_average = np.mean([

            item["score"]

            for item in results

        ])


        # ==========================================
        # DISPLAY RESULTS
        # ==========================================

        print("\n======================================")
        print("        LIVE VOICE RESULT")
        print("======================================")

        print("\nTop 3 Matches:")


        for item in top_3:

            print(

                item["name"],

                "->",

                round(item["score"], 3)

            )


        print("\n--------------------------------------")

        print(
            "Best similarity :",
            round(best_score, 3)
        )

        print(
            "Top 3 average   :",
            round(top3_average, 3)
        )

        print(
            "Overall average :",
            round(overall_average, 3)
        )

        print("--------------------------------------")


        # ==========================================
        # CHECK THREE CONDITIONS
        # ==========================================

        best_pass = (

            best_score >=
            BEST_SCORE_THRESHOLD

        )


        top3_pass = (

            top3_average >=
            TOP3_AVERAGE_THRESHOLD

        )


        overall_pass = (

            overall_average >=
            OVERALL_AVERAGE_THRESHOLD

        )


        # Count passed conditions
        passes = sum([

            best_pass,

            top3_pass,

            overall_pass

        ])


        # ==========================================
        # FINAL DECISION
        # ==========================================

        print("\nVerification checks:")

        print(
            "Best Score Check    :",
            "PASS" if best_pass else "FAIL"
        )

        print(
            "Top 3 Average Check :",
            "PASS" if top3_pass else "FAIL"
        )

        print(
            "Overall Average     :",
            "PASS" if overall_pass else "FAIL"
        )


        print("\n======================================")


        if passes >= 2:

            print("✅ VERIFIED: LIKELY REGISTERED USER")

        else:

            print("🚨 ALERT: UNKNOWN / UNVERIFIED VOICE")


        print("======================================\n")


except KeyboardInterrupt:

    print("\nVoiceGuard stopped.")