# setup_data.py
import kagglehub
import shutil
import os

def load_data():
    print("⬇️  Downloading dataset from Kaggle...")
    # Download latest version
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    
    print(f"✅  Download complete. Files located at: {path}")

    # Define target path
    target_dir = os.path.join("data", "raw")
    target_file = os.path.join(target_dir, "creditcard.csv")

    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Find the csv in the downloaded folder and move it
    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            source_file = os.path.join(path, filename)
            print(f"🚚  Moving {filename} to {target_dir}...")
            shutil.move(source_file, target_file)
            print(f"🎉  Success! Data is ready at: {target_file}")
            return

    print("❌  Error: No CSV file found in the downloaded package.")

if __name__ == "__main__":
    load_data()