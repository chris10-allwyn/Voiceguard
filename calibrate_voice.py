import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
import csv
import time


# ==========================================
# SETTINGS
# ==========================================

DURATION = 5
SAMPLE_RATE = 16000
TEMP_FILE = "temp_test.wav"

TESTS_PER_PERSON = 5

AUDIO_EXTENSIONS = [
    ".wav",
    ".ogg",
    ".mp3",
    ".m4a"
]


# ==========================================
# LOAD VOICE MODEL
# ==========================================

print("Loading voice model...")

encoder = VoiceEncoder()

print("Voice model loaded!")


# ==========================================
# LOAD 10 ENROLLED MICROPHONE RECORDINGS
# ==========================================

recordings_folder = Path("mic_recordings")

if not recordings_folder.exists():

    print("\nERROR: mic_recordings folder not found!")
    print("Run record_enrollment.py first.")

    exit()


recordings = sorted([
    file
    for file in recordings_folder.iterdir()
    if file.suffix.lower() in AUDIO_EXTENSIONS
])


print("\nFound", len(recordings), "registered microphone recordings.")


if len(recordings) == 0:

    print("ERROR: No enrollment recordings found!")

    exit()


# ==========================================
# CREATE REGISTERED EMBEDDINGS
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

        print("Error processing:", file.name)
        print(e)


if len(registered_embeddings) == 0:

    print("ERROR: Could not process enrollment recordings!")

    exit()


print("\n====================================")
print("VoiceGuard Calibration Ready")
print("Registered samples:", len(registered_embeddings))
print("====================================")


# ==========================================
# RECORD AND ANALYZE FUNCTION
# ==========================================

def record_and_analyze(person_name, test_number):

    print("\n------------------------------------")
    print(
        person_name,
        "- TEST",
        test_number,
        "of",
        TESTS_PER_PERSON
    )
    print("------------------------------------")

    input("\nPress ENTER when ready...")

    print("\n🎤 SPEAK NOW!")
    print("Recording for", DURATION, "seconds...")


    # ==========================================
    # RECORD AUDIO
    # ==========================================

    audio = sd.rec(

        int(DURATION * SAMPLE_RATE),

        samplerate=SAMPLE_RATE,

        channels=1,

        dtype="float32"

    )

    sd.wait()


    # Save temporary audio
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

        live_embedding = (
            live_embedding /
            np.linalg.norm(live_embedding)
        )


    except Exception as e:

        print("Error processing audio:", e)

        return None


    # ==========================================
    # COMPARE WITH ALL 10 REGISTERED SAMPLES
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


    top3_average = np.mean(

        [

            item["score"]

            for item in top_3

        ]

    )


    overall_average = np.mean(

        [

            item["score"]

            for item in results

        ]

    )


    # ==========================================
    # DISPLAY RESULT
    # ==========================================

    print("\n============== RESULT ==============")

    print(
        "Best Match:",
        top_3[0]["name"]
    )


    print(
        "Best Score:",
        round(best_score, 3)
    )


    print(
        "Top 3 Average:",
        round(top3_average, 3)
    )


    print(
        "Overall Average:",
        round(overall_average, 3)
    )


    print("====================================")


    return {

        "person": person_name,

        "test": test_number,

        "best_score": float(best_score),

        "top3_average": float(top3_average),

        "overall_average": float(overall_average)

    }


# ==========================================
# THREE PEOPLE TO TEST
# ==========================================

people = [

    "CHARVI",

    "OTHER_FEMALE",

    "OTHER_MALE"

]


all_results = []


# ==========================================
# RUN 5 TESTS FOR EACH PERSON
# ==========================================

for person in people:

    print("\n\n####################################")
    print("NOW TESTING:", person)
    print("####################################")

    print(
        "This person will give",
        TESTS_PER_PERSON,
        "voice samples."
    )


    for test_number in range(
        1,
        TESTS_PER_PERSON + 1
    ):

        result = record_and_analyze(

            person,

            test_number

        )


        if result is not None:

            all_results.append(result)


        time.sleep(1)


# ==========================================
# SAVE RESULTS TO CSV
# ==========================================

output_file = "mic_calibration_results.csv"


with open(

    output_file,

    "w",

    newline=""

) as file:

    writer = csv.DictWriter(

        file,

        fieldnames=[

            "person",

            "test",

            "best_score",

            "top3_average",

            "overall_average"

        ]

    )


    writer.writeheader()

    writer.writerows(all_results)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n\n====================================")
print("       FINAL CALIBRATION RESULTS")
print("====================================")


for person in people:


    person_results = [

        result

        for result in all_results

        if result["person"] == person

    ]


    if len(person_results) == 0:

        continue


    # Best scores
    best_scores = [

        result["best_score"]

        for result in person_results

    ]


    # Top 3 averages
    top3_scores = [

        result["top3_average"]

        for result in person_results

    ]


    # Overall averages
    overall_scores = [

        result["overall_average"]

        for result in person_results

    ]


    print("\n------------------------------------")

    print(person)


    print(
        "\nBest Scores:"
    )

    print(

        [

            round(score, 3)

            for score in best_scores

        ]

    )


    print(

        "Best Score Range:",

        round(min(best_scores), 3),

        "to",

        round(max(best_scores), 3)

    )


    print(

        "Average Best Score:",

        round(
            np.mean(best_scores),
            3
        )

    )


    print(
        "\nTop 3 Averages:"
    )

    print(

        [

            round(score, 3)

            for score in top3_scores

        ]

    )


    print(

        "Top 3 Range:",

        round(min(top3_scores), 3),

        "to",

        round(max(top3_scores), 3)

    )


    print(

        "Average Top 3:",

        round(
            np.mean(top3_scores),
            3
        )

    )


    print(
        "\nOverall Average:"
    )

    print(

        round(
            np.mean(overall_scores),
            3
        )

    )


print("\n====================================")
print("CALIBRATION COMPLETE!")
print("Results saved to:")
print(output_file)
print("====================================")