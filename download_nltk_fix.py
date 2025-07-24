import nltk
import os
import sys

# Define the EXACT path where you want NLTK data to go
# This should match the NLTK_DATA_PATH in your app.py
project_root = os.path.dirname(os.path.abspath(__file__))
target_nltk_path = os.path.join(project_root, 'nltk_data')

print(f"--- NLTK Downloader Script ---")
print(f"Target NLTK data path: {target_nltk_path}")

# Ensure the directory exists
if not os.path.exists(target_nltk_path):
    os.makedirs(target_nltk_path)
    print(f"Created directory: {target_nltk_path}")
else:
    print(f"Directory already exists: {target_nltk_path}")

# Add this path to NLTK's search paths
if target_nltk_path not in nltk.data.path:
    nltk.data.path.append(target_nltk_path)
    print(f"Added {target_nltk_path} to NLTK search paths for this session.")

# --- Attempt to download 'punkt_tab' ---
print("\nAttempting to download 'punkt_tab'...")
try:
    nltk.download('punkt_tab', download_dir=target_nltk_path)
    print("NLTK 'punkt_tab' downloaded successfully.")
except Exception as e:
    print(f"Error downloading 'punkt_tab': {e}")
    sys.exit(1) # Exit if critical download fails

# --- Verification (for both punkt and stopwords, just to be thorough) ---
print("\n--- Verifying Downloads ---")

# Verify 'punkt' (the general one)
try:
    nltk.data.find('tokenizers/punkt')
    print("SUCCESS: General 'punkt' data found by NLTK!")
except nltk.downloader.DownloadError:
    print("FAILURE: General 'punkt' data NOT found. Please check internet/permissions.")
    sys.exit(1)

# Verify 'punkt_tab/english' (the specific one causing issues)
try:
    nltk.data.find('tokenizers/punkt_tab/english/')
    print("SUCCESS: 'punkt_tab/english/' data found by NLTK!")
except nltk.downloader.DownloadError:
    print("FAILURE: 'punkt_tab/english/' data NOT found by NLTK. This is unexpected after download.")
    print("Please check your 'nltk_data/tokenizers/punkt_tab/english/' folder manually.")
    sys.exit(1)

# Verify 'stopwords'
try:
    nltk.data.find('corpora/stopwords')
    print("SUCCESS: 'stopwords' data found by NLTK!")
except nltk.downloader.DownloadError:
    print("FAILURE: 'stopwords' data NOT found. Please check internet/permissions.")
    sys.exit(1)

print("\nAll required NLTK downloads and verifications complete. You can now run your Flask app.")