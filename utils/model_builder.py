import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import joblib
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    roc_auc_score
)

def create_cnn_model(input_shape, num_classes):
    print(f"input shape : {input_shape}")
    print(f"number of classes : {num_classes}")
    model = Sequential([
        Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_model(X_train, y_train, X_val, y_val, input_shape, num_classes, save_path):
    model = create_cnn_model(input_shape, num_classes)
    
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    model.summary()
    # Save the model
    os.makedirs("models", exist_ok=True)
    model.save_weights(os.path.join("models", "cnn_model_weights.h5"))
    model.save_weights(os.path.join("models", "cnn_model_weights.keras"))
    model.save("models/cnn_model.h5")
    model.save("models/cnn_model.keras")
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plot_model(model,to_file="plots/cnn_model_architecture.png", show_layer_names=True,expand_nested=True,dpi=100)

    best_epoch = int(np.argmin(history.history['val_loss']))
    best_val_loss = history.history['val_loss'][best_epoch]
    best_val_acc = history.history['val_accuracy'][best_epoch]
    print(f"Best epoch: {best_epoch + 1}, val_loss={best_val_loss:.4f}, val_accuracy={best_val_acc:.4f}")
    return model, history

def train_ml_models(X_train, y_train, save_dir="models"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    models = {
        'svm': SVC(probability=True),
        'rf': RandomForestClassifier(),
        'dt': DecisionTreeClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(save_dir, f"{name}_model.joblib"))

    return models

def save_history_to_csv(history, filepath="models/training_history.csv"):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pd.DataFrame(history.history).to_csv(filepath, index_label="epoch")


def evaluate_models(models, X_val, y_val, save_path="models/comparison_metrics.csv"):
    rows = []
    for name, model in models.items():
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val) 

        row = {
            "model": name,
            "accuracy": accuracy_score(y_val, preds),
            "precision_macro": precision_score(y_val, preds, average="macro"),
            "recall_macro": recall_score(y_val, preds, average="macro"),
            "f1_macro": f1_score(y_val, preds, average="macro"),
        }

        # ROC-AUC needs special handling for multiclass vs binary
        if len(np.unique(y_val)) == 2:
            row["roc_auc"] = roc_auc_score(y_val, probs[:, 1])
        else:
            row["roc_auc_ovr"] = roc_auc_score(y_val, probs, multi_class="ovr", average="macro")

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df

def per_class_report(models, X_val, y_val, save_dir="models"):
    for name, model in models.items():
        preds = model.predict(X_val)
        report = classification_report(y_val, preds, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(f"{save_dir}/{name}_classification_report.csv", index=False)