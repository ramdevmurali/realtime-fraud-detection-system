import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
import os
from src.data_loader import load_and_preprocess_data
from src.model_builder import build_model
from src.utils import load_config, configure_logger

configure_logger()

def train_model():
    config = load_config()
    
    # 1. Load Data
    logging.info("Starting data loading...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()

    # 2. Build Model
    logging.info("Building model architecture...")
    model = build_model()

    # 3. Define Callbacks
    save_path = config['model']['save_path']
    patience = config['training']['patience']
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    ]

    # 4. Train
    logging.info("Starting training...")
    batch_size = config['model']['batch_size']
    epochs = config['model']['epochs']

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Final Evaluation
    logging.info("Evaluating on Test Set...")
    results = model.evaluate(X_test, y_test, verbose=0)
    
    metrics = {k: v for k, v in zip(model.metrics_names, results)}
    logging.info(f"Test Results: {metrics}")
    
    logging.info(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    train_model()