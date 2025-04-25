import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_wav2vec2_features(audio_data):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = []
    
    for audio in audio_data:
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}  # Move to GPU if available
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        features.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())  # Take mean across time
        
    return np.array(features)

def extract_features(audio_data):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to("cuda" if torch.cuda.is_available() else "cpu")
    device = model.device
    features = []
    for audio in audio_data:
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        feat = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        features.append(feat)
    return np.vstack(features)