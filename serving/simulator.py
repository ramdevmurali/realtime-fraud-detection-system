import pandas as pd
import requests
import time
import random
import logging
import sys
import os

# Add root to path to load config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_config, configure_logger

configure_logger()
config = load_config()

API_URL = "http://localhost:5001/predict"

def simulate_traffic():
    """
    Simulates real-time transaction traffic by sending random
    rows from the dataset to the API.
    """
    raw_path = config['data']['raw_file']
    logging.info(f"Loading data from {raw_path} for simulation...")
    
    # Load data (we assume this simulates the 'incoming stream')
    df = pd.read_csv(raw_path)
    
    # Drop the class label (we don't send the answer to the model!)
    features = df.drop('Class', axis=1)
    
    logging.info("🚀 Starting traffic simulation... Press Ctrl+C to stop.")
    
    try:
        while True:
            # Pick a random row
            random_index = random.randint(0, len(features) - 1)
            row = features.iloc[random_index]
            
            # Convert to dictionary
            payload = row.to_dict()
            
            # Send Request
            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    status = "🔴 FRAUD DETECTED" if result['alert'] else "🟢 Normal"
                    logging.info(f"Sent ID {random_index} | Score: {result['fraud_score']:.4f} | {status}")
                else:
                    logging.error(f"API Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                logging.error("Could not connect to API. Is it running?")
                time.sleep(5)
            
            # Random sleep to mimic traffic patterns (0.1s to 2s)
            time.sleep(random.uniform(0.1, 1.0))

    except KeyboardInterrupt:
        logging.info("Simulation stopped.")

if __name__ == "__main__":
    simulate_traffic()