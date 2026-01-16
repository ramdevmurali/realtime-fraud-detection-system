import sys
import os
import joblib
import pandas as pd
import sqlite3
import datetime
import logging
from flask import Flask, request, jsonify
from keras.models import load_model

# Add root to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_config, configure_logger

app = Flask(__name__)
configure_logger()
config = load_config()

# Global variables
model = None
scaler = None
DB_PATH = config['data'].get('db_path', 'data/database.db')

def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            input_data TEXT,
            prediction_score REAL,
            is_fraud INTEGER,
            latency_ms REAL
        )
    ''')
    conn.commit()
    conn.close()
    logging.info("Database initialized.")

def load_artifacts():
    """Load model and scaler."""
    global model, scaler
    
    # Load Scaler
    scaler_path = config['data']['scaler_path']
    # Check if file exists
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Did you run training?")
    scaler = joblib.load(scaler_path)
    logging.info(f"Scaler loaded from {scaler_path}")

    # Load Model
    model_path = config['model']['save_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Did you run training?")
    model = load_model(model_path)
    logging.info(f"Model loaded from {model_path}")

# Initialize everything on startup
with app.app_context():
    init_db()
    load_artifacts()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = datetime.datetime.now()
        data = request.json
        
        # Expecting input like: {"V1": 0.1, "V2": -0.5, ...}
        # We need to ensure the columns are in the correct order as training
        # For simplicity in this demo, we convert values to list, but in prod 
        # you would enforce schema validation.
        
        input_df = pd.DataFrame([data])
        
        # Preprocess
        scaled_data = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(scaled_data, verbose=0)
        fraud_score = float(prediction[0][0])
        is_fraud = 1 if fraud_score > config['training']['threshold'] else 0
        
        # Log to DB
        end_time = datetime.datetime.now()
        latency = (end_time - start_time).total_seconds() * 1000
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (timestamp, input_data, prediction_score, is_fraud, latency_ms) VALUES (?, ?, ?, ?, ?)",
            (start_time, str(data), fraud_score, is_fraud, latency)
        )
        conn.commit()
        conn.close()

        return jsonify({
            'fraud_score': fraud_score,
            'is_fraud': bool(is_fraud),
            'alert': bool(is_fraud) 
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Disable debug mode to prevent tensorflow reloading issues on M1
    app.run(host='0.0.0.0', port=5001, debug=False)