from flask import Flask, request, jsonify, render_template
import os
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import logging
import sys # Import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- NLTK Data Management ---
# Define the path where NLTK data will be stored within your project
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), 'nltk_data')

# Ensure the NLTK data path exists and is added to NLTK's search paths
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)
    logging.info(f"Created NLTK data directory: {NLTK_DATA_PATH}")

if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)
    logging.info(f"Added {NLTK_DATA_PATH} to NLTK data search paths.")

# Function to explicitly load Punkt (and stopwords) or download if missing
def load_nltk_corpora_robustly():
    required_corpora = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords'
    }

    for corpus_name, data_path_key in required_corpora.items():
        try:
            # Try to find the data using NLTK's standard mechanism
            nltk.data.find(data_path_key)
            logging.info(f"NLTK corpus '{corpus_name}' found successfully via standard lookup.")
        except nltk.downloader.DownloadError:
            # If not found, attempt to download it to our specific project path
            logging.warning(f"NLTK corpus '{corpus_name}' not found. Attempting download to {NLTK_DATA_PATH}...")
            try:
                nltk.download(corpus_name, download_dir=NLTK_DATA_PATH)
                logging.info(f"NLTK corpus '{corpus_name}' downloaded successfully to {NLTK_DATA_PATH}.")
            except Exception as e:
                logging.error(f"Failed to download NLTK corpus '{corpus_name}': {e}", exc_info=True)
                sys.exit(1) # Exit application if critical data cannot be downloaded

    # --- FINAL ROBUST VERIFICATION for punkt specifically ---
    # This is a fallback if `nltk.data.find` still somehow fails but files exist
    punkt_file_path = os.path.join(NLTK_DATA_PATH, 'tokenizers', 'punkt', 'english.pickle')
    if not os.path.exists(punkt_file_path):
        logging.critical(f"CRITICAL: 'punkt/english.pickle' not found at expected location: {punkt_file_path}")
        logging.critical("This indicates a persistent NLTK data issue. Summarization will likely fail.")
        # If it truly cannot be found, you might want to exit here as well,
        # or just log and let the summarization function fail gracefully.
        # For now, we'll just log critically.
    else:
        logging.info(f"Confirmed 'punkt/english.pickle' exists at {punkt_file_path}")


# Call the robust loading function when the app starts
load_nltk_corpora_robustly()

# --- Summarization Function ---
def summarize_text_backend(text, sentence_count=3):
    """Summarizes the given text using LSA algorithm."""
    if not text or len(text.strip()) == 0:
        logging.warning("Received empty text for summarization.")
        return None

    try:
        # Use NLTK's sentence tokenizer (which relies on punkt)
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= sentence_count and len(text.split()) < 50:
            logging.info(f"Text too short to effectively summarize ({len(sentences)} sentences). Returning original text.")
            return text.strip()

        # Sumy requires a PlaintextParser and Tokenizer
        parser = PlaintextParser.from_string(text, Tokenizer("english")) # Tokenizer uses NLTK internally
        
        # Initialize stemmer and summarizer
        stemmer = Stemmer("english")
        summarizer = LsaSummarizer(stemmer)
        
        # Set stop words for the summarizer
        # This will also rely on NLTK finding the 'stopwords' corpus
        summarizer.stop_words = get_stop_words("english")

        # Perform summarization
        summary = summarizer(parser.document, sentence_count)
        summarized_result = " ".join([str(sentence) for sentence in summary])
        
        logging.info(f"Text summarized to {len(summarized_result.split())} words from {len(text.split())} words.")
        return summarized_result
    except Exception as e:
        logging.error(f"Error during summarization: {e}", exc_info=True)
        return None

# --- Flask Routes ---

@app.route('/')
def index():
    """Home page with links to other modules."""
    return render_template('index.html')

@app.route('/summarizer')
def summarizer_page():
    """Text Summarizer module page."""
    return render_template('summarizer.html')

@app.route('/voice_to_text')
def voice_to_text_page():
    """Voice to Text module page."""
    return render_template('voice_to_text.html')

@app.route('/text_to_voice')
def text_to_voice_page():
    """Text to Voice module page."""
    return render_template('text_to_voice.html')

@app.route('/api/summarize', methods=['POST'])
def api_summarize_endpoint():
    """API endpoint to receive text, summarize it, and return the summary."""
    data = request.get_json()
    if not data or 'text' not in data:
        logging.warning("Bad request: Missing 'text' in JSON payload for summarization API.")
        return jsonify({"error": "Missing 'text' in request"}), 400

    input_text = data['text']
    sentence_count = data.get('sentences', 3) # Default to 3 sentences

    logging.info(f"API: Received text for summarization (length: {len(input_text)}). Sentences requested: {sentence_count}")

    summarized_text = summarize_text_backend(input_text, sentence_count=sentence_count)

    if summarized_text is not None:
        return jsonify({"summary": summarized_text})
    else:
        logging.error("API: Failed to generate summary for the provided text.")
        return jsonify({"error": "Could not summarize text. Check server logs for details."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)