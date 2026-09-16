import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from pathlib import Path

# =========================
# SETTINGS
# =========================

SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
EPOCHS = 10
BATCH_SIZE = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATASET
# =========================

class VoiceDataset(Dataset):

    def __init__(self):

        self.files = []

        real_folder = Path("dataset/real")
        ai_folder = Path("dataset/ai")

        for file in real_folder.iterdir():
            if file.suffix.lower() in [".wav", ".mp3", ".ogg", ".m4a"]:
                self.files.append((file, 0))

        for file in ai_folder.iterdir():
            if file.suffix.lower() in [".wav", ".mp3", ".ogg", ".m4a"]:
                self.files.append((file, 1))

        print("Real files:", len(list(real_folder.iterdir())))
        print("AI files:", len(list(ai_folder.iterdir())))
        print("Total files:", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file, label = self.files[index]

        audio, _ = librosa.load(
            file,
            sr=SAMPLE_RATE,
            duration=DURATION
        )

        # Make every audio exactly 5 seconds
        required_length = SAMPLE_RATE * DURATION

        if len(audio) < required_length:
            audio = np.pad(
                audio,
                (0, required_length - len(audio))
            )
        else:
            audio = audio[:required_length]

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS
        )

        mel = librosa.power_to_db(mel, ref=np.max)

        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # CNN expects channel dimension
        mel = torch.tensor(
            mel,
            dtype=torch.float32
        ).unsqueeze(0)

        return mel, torch.tensor(
            label,
            dtype=torch.float32
        )


# =========================
# CNN MODEL
# =========================

class VoiceDetectorCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        x = self.network(x)
        x = self.classifier(x)

        return x.squeeze(1)


# =========================
# TRAIN
# =========================

dataset = VoiceDataset()

if len(dataset) == 0:
    print("ERROR: No audio files found.")
    exit()

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = VoiceDetectorCNN().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("\nTraining on:", DEVICE)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for audio, labels in loader:

        audio = audio.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        output = model(audio)

        loss = criterion(
            output,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Loss: {total_loss / len(loader):.4f}"
    )


# =========================
# SAVE MODEL
# =========================

Path("models").mkdir(
    exist_ok=True
)

torch.save(
    model.state_dict(),
    "models/ai_voice_detector.pth"
)

print("\n================================")
print("TRAINING COMPLETE")
print("Model saved:")
print("models/ai_voice_detector.pth")
print("================================")