import os
import sys

# MAC OS SILICON FIXES -------------------------
# This specific flag fixes the "mutex lock failed" error on M1/M2/M3
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# Force single-threaded operation for OpenMP to avoid conflicts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ----------------------------------------------

from src.trainer import train_model

if __name__ == "__main__":
    print("🚀 Starting Training Pipeline...")
    train_model()
    print("✨ Pipeline Complete.")