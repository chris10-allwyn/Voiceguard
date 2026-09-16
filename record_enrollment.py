import sounddevice as sd
import soundfile as sf
from pathlib import Path

DURATION = 5
SAMPLE_RATE = 16000
NUM_RECORDINGS = 10

output_folder = Path("mic_recordings")
output_folder.mkdir(exist_ok=True)

print("====================================")
print("VOICE ENROLLMENT")
print("====================================")
print("We will record", NUM_RECORDINGS, "samples.")
print("Each recording is", DURATION, "seconds.")
print("Use the SAME microphone you will use")
print("for live verification.")
print("====================================")

for i in range(1, NUM_RECORDINGS + 1):

    input(f"\nPress ENTER for recording {i}/{NUM_RECORDINGS}...")

    print("🎤 Speak normally!")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    filename = output_folder / f"charvi_mic_{i}.wav"

    sf.write(
        filename,
        audio,
        SAMPLE_RATE
    )

    print("✅ Saved:", filename)

print("\n====================================")
print("Enrollment complete!")
print("Saved recordings:", NUM_RECORDINGS)
print("Folder:", output_folder)
print("====================================")