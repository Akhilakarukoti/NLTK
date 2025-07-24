import nltk
import os
import sys # Import sys to access command line arguments for a cleaner exit

# Define the EXACT path where you want NLTK data to go
# This should match the NLTK_DATA_PATH in your app.py
# os.path.dirname(__file__) gets the directory of this script (ml folder)
target_nltk_path = os.path.join(os.path.dirname(__file__), 'nltk_data')

print(f"--- NLTK Downloader Script ---")
print(f"Target NLTK data path: {target_nltk_path}")

# Ensure the directory exists
if not os.path.exists(target_nltk_path):
    os.makedirs(target_nltk_path)
    print(f"Created directory: {target_nltk_path}")
else:
    print(f"Directory already exists: {target_nltk_path}")

# Add this path to NLTK's search paths (important for it to be found later)
if target_nltk_path not in nltk.data.path:
    nltk.data.path.append(target_nltk_path)
    print(f"Added {target_nltk_path} to NLTK search paths.")

# --- Attempt to download 'punkt' ---
print("\nAttempting to download 'punkt'...")
try:
    # The download_dir argument forces it to download to this specific path
    nltk.download('punkt', download_dir=target_nltk_path)
    print("NLTK 'punkt' download attempt complete.")
except Exception as e:
    print(f"Error downloading 'punkt': {e}")
    sys.exit(1) # Exit with an error code if download fails

# --- Attempt to download 'stopwords' ---
print("\nAttempting to download 'stopwords'...")
try:
    # The download_dir argument forces it to download to this specific path
    nltk.download('stopwords', download_dir=target_nltk_path)
    print("NLTK 'stopwords' download attempt complete.")
except Exception as e:
    print(f"Error downloading 'stopwords': {e}")
    sys.exit(1) # Exit with an error code if download fails

# --- VERIFICATION ---
print("\n--- Verifying Downloads ---")
try:
    nltk.data.find('tokenizers/punkt')
    print("SUCCESS: 'punkt' data found by NLTK!")
except nltk.downloader.DownloadError:
    print("FAILURE: 'punkt' data NOT found by NLTK after download. Please check your internet connection and folder permissions.")
    sys.exit(1)

try:
    nltk.data.find('corpora/stopwords')
    print("SUCCESS: 'stopwords' data found by NLTK!")
except nltk.downloader.DownloadError:
    print("FAILURE: 'stopwords' data NOT found by NLTK after download. Please check your internet connection and folder permissions.")
    sys.exit(1)

print("\nNLTK downloads and verification complete. You can now run your Flask app.")