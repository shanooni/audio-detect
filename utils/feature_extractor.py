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
    Returns:
        features: NumPy array of shape (N, 768)
        kept_indices: list of original indices that survived extraction
                      (use this to filter labels so they stay aligned)
    """

    features = []
    kept_indices = []

    for i, audio in enumerate(audio_data):
        if audio is None or len(audio) < 400:  # ~25ms at 16kHz, safe minimum above conv kernel size
            print(f"Skipping sample {i}: invalid/too-short audio "
                  f"({0 if audio is None else len(audio)} samples)")
            continue

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
        kept_indices.append(i)

    if len(features) == 0:
        raise RuntimeError("No valid features extracted — all audio inputs were empty or too short")

    return np.vstack(features), kept_indices


def save_features_to_csv(features, labels, filepath):
    df = pd.DataFrame(features)
    df["label"] = labels
    df.to_csv(filepath, index=False)