import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import joblib
from tensorflow.keras.models import load_model

# Load models once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

# Load ensemble model components
svm_model = joblib.load("models/svm_model.joblib")
dt_model = joblib.load("models/dt_model.joblib")
rf_model = joblib.load("models/rf_model.joblib")
cnn_model = load_model("models/cnn_model.h5")

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = wav2vec_model(**inputs)

        feat = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return feat
    except Exception as e:
        print("Error extracting features with Wav2Vec2:", e)
        return None

def predict_audio_deepfake(file_path):
    features = extract_features(file_path)
    if features is None:
        return None

    # Predict probabilities with individual models
    prob_svm = svm_model.predict_proba(features)[0]
    prob_dt = dt_model.predict_proba(features)[0]
    prob_rf = rf_model.predict_proba(features)[0]

    # CNN model expects 3D input
    features_cnn = features.reshape(-1, 24, 32, 1)
    prob_cnn = cnn_model.predict(features_cnn, verbose=0)[0]

    # Weighted ensemble
    avg_prob = (
        0.4 * prob_cnn +
        0.2 * prob_svm +
        0.2 * prob_dt +
        0.2 * prob_rf
    )

    predicted_class = np.argmax(avg_prob)
    return "real" if predicted_class == 0 else "deepfake"
