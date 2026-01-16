import yaml
import logging

def load_config(config_path="config/config.yaml"):
    """Load configuration from a YAML file."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        raise

def configure_logger():
    """Set up logging to console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/system.log"),
            logging.StreamHandler()
        ]
    )