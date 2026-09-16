import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav

print("Loading voice encoder...")
encoder = VoiceEncoder()

folder = Path("recordings")

audio_files = [
    f for f in folder.iterdir()
    if f.suffix.lower() in [".wav", ".ogg", ".mp3", ".m4a"]
]

embeddings = {}
audio_files = sorted(audio_files)

# Create embedding for each recording
for file in audio_files:
    print("Processing:", file.name)

    wav = preprocess_wav(file)
    embedding = encoder.embed_utterance(wav)

    embedding = embedding / np.linalg.norm(embedding)

    embeddings[file.name] = embedding


print("\n========================================")
print("SIMILARITY BETWEEN REGISTERED RECORDINGS")
print("========================================\n")

average_scores = {}

for name1, emb1 in embeddings.items():

    scores = []

    for name2, emb2 in embeddings.items():

        if name1 == name2:
            continue

        similarity = float(np.dot(emb1, emb2))
        scores.append(similarity)

    average_scores[name1] = np.mean(scores)

    print(
        f"{name1:25} "
        f"Average similarity: {average_scores[name1]:.3f}"
    )


print("\n========================================")
print("RECORDINGS RANKED")
print("========================================")

ranked = sorted(
    average_scores.items(),
    key=lambda x: x[1],
    reverse=True
)

for name, score in ranked:
    print(f"{name:25} -> {score:.3f}")


print("\n========================================")
print("POSSIBLE OUTLIERS")
print("========================================")

overall_mean = np.mean(list(average_scores.values()))

for name, score in average_scores.items():

    if score < overall_mean - 0.05:
        print(name, "-> Possible outlier:", round(score, 3))

print("\nOverall mean:", round(overall_mean, 3))