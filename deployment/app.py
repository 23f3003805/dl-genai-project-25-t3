import gradio as gr
import torch
import numpy as np
import librosa
import json

from model import SimpleCNN, CRNN, EfficientNetAudio, AudioClassifier, GENRES


with open("label_map.json") as f:
    raw = json.load(f)
IDX_TO_GENRE = {int(k): v for k, v in raw.items()}


device = torch.device("cpu")  


def load_model(cls, path):
    m = cls().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

cnn_model = load_model(SimpleCNN, "cnn.pth")
crnn_model = load_model(CRNN,"crnn.pth")
effnet_model = load_model(EfficientNetAudio, "effnet.pth")
effnet_aug_model = load_model(AudioClassifier, "effnet_aug.pth")


def random_crop(y, crop_size):
    if len(y) <= crop_size:
        return np.pad(y, (0, crop_size - len(y)))
    start = np.random.randint(0, len(y) - crop_size)
    return y[start:start + crop_size]


def predict_audio(audio_path):
    y, _ = librosa.load(audio_path, sr=22050)

    crop_size = 22050 * 6
    step = crop_size // 4
    crops = []
    for start in range(0, max(1, len(y) - crop_size + 1), step):
        crops.append(y[start:start + crop_size])
    if not crops:
        crops.append(random_crop(y, crop_size))

    all_preds = []
    for crop in crops:
        x = torch.tensor(crop).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            p1 = torch.softmax(cnn_model(x), 1)
            p2 = torch.softmax(crnn_model(x), 1)
            p3 = torch.softmax(effnet_model(x), 1)
            p4 = torch.softmax(effnet_aug_model(x), 1)
            p  = (0.15*p1 + 0.25*p2 + 0.35*p3 + 0.25*p4).cpu().numpy()
        all_preds.append(p[0])

    avg = np.mean(all_preds, axis=0)

    return {IDX_TO_GENRE[i]: float(avg[i]) for i in range(len(GENRES))}


demo = gr.Interface(
    fn=predict_audio,
    inputs=gr.Audio(type="filepath", label="Upload a music clip (.wav or .mp3)"),
    outputs=gr.Label(num_top_classes=5, label="Genre Prediction"),
    title="🎵 Music Genre Classifier",
    description=(
        "Upload a music clip and the ensemble model (CNN + CRNN + EfficientNet + EfficientNetAugmented) "
        "will predict its genre. Trained on 10 genres: "
        + ", ".join(GENRES)
    ),
    examples=[],         
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()