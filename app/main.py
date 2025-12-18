import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.feature_extractor import extract_wav2vec2_features
from utils.data_loader import load_audio_files, prepare_data
from utils.model_builder import train_ml_models, train_cnn_model
from utils.model_trainer import ensemble_predict
from utils.helper import evaluate_performance
from visualizer import visualizer


def run_pipeline(data_dir):
    audio_data, labels = load_audio_files(data_dir)
    features = extract_wav2vec2_features(audio_data)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    feature_reshape = np.array(features).reshape(-1, 24, 32, 1)
    X_train, X_test, y_train, y_test = train_test_split(feature_reshape, encoded_labels, test_size=0.2, random_state=42,  stratify = labels)

    # Classical models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    ml_models = train_ml_models(X_train_flat, y_train)

    # CNN
    # Reshape the feature vectors for CNN input
    # try:
    #     X_train_cnn = X_train.reshape(-1, 24, 32, 1)
    #     X_test_cnn = X_test.reshape(-1, 24, 32, 1)
    # except Exception as e:
    #     print("Error reshaping features for CNN:", e)
    #     return
    input_shape = (24, 32, 1)
    num_classes = len(np.unique(labels))
    cnn_model, history = train_cnn_model(X_train, y_train, X_test, y_test, input_shape, num_classes,"models")
    with open('config/weights.json', 'r') as f:
        weights = json.load(f)['weights']
    
    # Ensemble predictions
    # ensemble_preds = ensemble_predict(X_test_flat,input_shape, ml_models, cnn_model,weights )
    ensemble_preds = ensemble_predict(X_test_flat, X_test, ml_models, cnn_model, weights)

    metrics = evaluate_performance(y_test, ensemble_preds)
    visualizer.plot_metrics(metrics, save_path="plots/metrics_plot.png")
    class_names = ['real', 'fake']
    visualizer.plot_confusion_matrix(metrics["confusion_matrix"], class_labels=class_names, save_path="plots/conf_matrix.png")


# ---- Run ----
if __name__ == "__main__":
    run_pipeline("/Users/shanoonissaka/Documents/school/thesis-project/datasets/audio/training")  # Change to your audio training directory
