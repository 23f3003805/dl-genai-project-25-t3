import torch
import torch.nn as nn
import torchaudio
import timm
import random

GENRES = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050,
    n_fft=1024,
    hop_length=256,
    n_mels=128,
    f_min=20,
    f_max=11025
)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = mel_transform
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, len(GENRES))

    def forward(self, x):
        x = self.mel(x)
        x = torch.log(x + 1e-6)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = mel_transform
        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.gru = nn.GRU(
            input_size=128, 
            hidden_size=128, 
            num_layers=2,
            batch_first=True, 
            bidirectional=True)
        self.fc = nn.Linear(256, len(GENRES))

    def forward(self, x):
        x = self.mel(x)
        x = torch.log(x + 1e-6)
        x = self.cnn(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        return self.fc(x)


class EfficientNetAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = mel_transform
        self.freq_mask = torchaudio.transforms.FrequencyMasking(24)
        self.time_mask = torchaudio.transforms.TimeMasking(40)
        self.backbone = timm.create_model(
            "tf_efficientnet_b0", pretrained=False,
            in_chans=1, num_classes=len(GENRES)
        )

    def forward(self, x):
        x = self.mel(x)
        x = torch.log(x + 1e-6)
        x = (x - x.mean(dim=(2,3), keepdim=True)) / (x.std(dim=(2,3), keepdim=True) + 1e-6)
        return self.backbone(x)


class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = mel_transform
        self.freq_mask = torchaudio.transforms.FrequencyMasking(48)
        self.time_mask = torchaudio.transforms.TimeMasking(96)
        self.backbone = timm.create_model(
            "tf_efficientnet_b0", pretrained=False,
            in_chans=1, num_classes=len(GENRES)
        )

    def forward(self, x):
        x = self.mel(x)
        x = torch.log(x + 1e-6)
        x = (x - x.mean(dim=(2,3), keepdim=True)) / (x.std(dim=(2,3), keepdim=True) + 1e-6)
        return self.backbone(x)