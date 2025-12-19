import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import joblib
import os


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
    plot_model(model,to_file="plots/cnn_model_architecture.png", show_layer_names=True,expand_nested=True,dpi=100)
    return model, history

def train_ml_models(X_train, y_train, save_dir="models"):
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