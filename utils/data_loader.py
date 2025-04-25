from sklearn.model_selection import train_test_split
import os
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

# Define a function to load audio and extract Wav2Vec2 features
def load_audio_files(directory, sr=16000):
    audio_data = []
    labels = []
    class_labels = {label: i for i, label in enumerate(os.listdir(directory))}

    for label in os.listdir(directory):
        class_dir = os.path.join(directory, label)
        if os.path.isdir(class_dir) and class_dir != ".DS_Store":
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                try:
                    audio, _ = librosa.load(file_path, sr=sr)
                    audio_data.append(audio)
                    labels.append(class_labels[label])
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    return np.array(audio_data, dtype=object), np.array(labels)

def prepare_data(flatten_features, flatten_labels):
    X_train, X_val, y_train, y_val = train_test_split(flatten_features, flatten_labels, test_size=0.2, random_state=42)
    
    # Reshape for Conv2D: (samples, height, width, channels)
    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1))
    X_val_cnn = X_val.reshape((X_val.shape[0], X_val.shape[1], 1, 1))

    return X_train, X_val, X_train_cnn, X_val_cnn, y_train, y_val