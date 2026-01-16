import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from src.utils import load_config

def build_model():
    config = load_config()
    input_dim = config['model']['input_dim']
    learning_rate = config['model']['learning_rate']

    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy', 
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]
    )

    return model