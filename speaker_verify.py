import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav

print("Loading pretrained voice model...")

encoder = VoiceEncoder()


# ==========================================
# CREATE VOICE PROFILE FROM ALL RECORDINGS
# ==========================================

def create_voice_profile(folder="recordings"):

    folder = Path(folder)

    # Check if recordings folder exists
    if not folder.exists():
        print("ERROR: Recordings folder not found!")
        print("Expected location:", folder.absolute())
        return

    embeddings = []

    print("\nSearching for recordings...\n")

    # Read all files inside recordings folder
    for file in folder.iterdir():

        # Accept audio files
        if file.suffix.lower() not in [".wav", ".ogg", ".mp3", ".m4a"]:
            continue

        print("Processing:", file.name)

        try:
            # Convert audio to waveform
            wav = preprocess_wav(file)

            # Generate voice embedding
            embedding = encoder.embed_utterance(wav)

            # Store embedding
            embeddings.append(embedding)

            print("Done!")

        except Exception as e:
            print("Skipping:", file.name)
            print("Error:", e)

    # Check if any recordings were processed
    if len(embeddings) == 0:
        print("\nERROR: No audio files could be processed!")
        return

    # ==========================================
    # CREATE AVERAGE VOICE PROFILE
    # ==========================================

    print("\nCreating average voice profile...")

    profile = np.mean(embeddings, axis=0)

    # Normalize profile
    profile = profile / np.linalg.norm(profile)

    # Create models folder
    models_folder = Path("models")
    models_folder.mkdir(exist_ok=True)

    # Save voice profile
    np.save(
        models_folder / "user_profile.npy",
        profile
    )

    print("\n===================================")
    print("SUCCESS!")
    print("Processed recordings:", len(embeddings))
    print("Voice profile created!")
    print("Saved as: models/user_profile.npy")
    print("===================================")


# ==========================================
# COMPARE LIVE VOICE WITH USER PROFILE
# ==========================================

def compare_voice(audio_path):

    # Load registered user voice profile
    profile_path = Path("models/user_profile.npy")

    if not profile_path.exists():
        raise FileNotFoundError(
            "Voice profile not found! Run create_profile.py first."
        )

    user_profile = np.load(profile_path)

    # Process incoming audio
    print("Analyzing live voice...")

    wav = preprocess_wav(Path(audio_path))

    # Generate embedding for incoming voice
    live_embedding = encoder.embed_utterance(wav)

    # Normalize embedding
    live_embedding = (
        live_embedding /
        np.linalg.norm(live_embedding)
    )

    # Calculate cosine similarity
    similarity = np.dot(
        user_profile,
        live_embedding
    )

    # Convert to Python float
    return float(similarity)