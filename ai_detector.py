import torch
import torch.nn as nn
import librosa
import numpy as np
from pathlib import Path


SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# MODEL
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
# LOAD MODEL
# =========================

model = VoiceDetectorCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        "models/ai_voice_detector.pth",
        map_location=DEVICE
    )
)

model.eval()

print("AI voice detector loaded.")


# =========================
# DETECTION FUNCTION
# =========================

def detect_ai_voice(audio_file):

    audio, _ = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

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

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel = (
        mel - mel.mean()
    ) / (
        mel.std() + 1e-6
    )


    tensor = torch.tensor(
        mel,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(DEVICE)


    # Prediction

    with torch.no_grad():

        output = model(tensor)

        probability = torch.sigmoid(
            output
        ).item()


    ai_percentage = probability * 100

    if probability >= 0.5:

        result = "AI-GENERATED"

    else:

        result = "REAL"


    return result, ai_percentage