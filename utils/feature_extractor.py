import torch
import numpy as np
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load once (important for performance)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
model.eval()


def extract_wav2vec2_features(audio_data, sampling_rate=16000):
    """
    Extract Wav2Vec2 embeddings (mean-pooled) for a list of raw audio arrays.
    Returns a NumPy array of shape (N, 768)
    """

    features = []

    for audio in audio_data:
        # Convert audio to model input
        inputs = processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        )

        # Move tensors to GPU if available
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Take mean over time dimension → (1, 768)
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        features.append(emb)

    return np.vstack(features)


    def save_features_to_csv(features, labels, filepath):
        df = pd.DataFrame(features)
        df["label"] = labels
        df.to_csv(filepath, index=False)