import asyncio, difflib, time #multiprocessser good for sockets
start_time = time.time()
print("importing libraries")

import websockets , json # the json is bc we're sending a json object from the website
import wave , base64
import os #library for anything that has to do with your hard-drive
end_time = time.time()
print(f"Time taken to import libraries: {end_time - start_time} seconds")
print("importing torch")
start_time = time.time()
import torch, sqlite3
end_time = time.time()
print(f"Time taken to import torch and sqlite3: {end_time - start_time} seconds")
print("importing librosa")
start_time = time.time()
import librosa #library to analyse and process audio . soundfile is similar 
end_time = time.time()
print(f"Time taken to import librosa: {end_time - start_time} seconds")
print("importing transformers")
start_time = time.time()
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
end_time = time.time()
print(f"Time taken to import transformers: {end_time - start_time} seconds")
print("importing ssl")
start_time = time.time()
import ssl  # Add this import at the top
end_time = time.time()
print(f"Time taken to import ssl: {end_time - start_time} seconds")
print("importing secrets")
start_time = time.time()
import secrets 
end_time = time.time()
print(f"Time taken to import secrets: {end_time - start_time} seconds")
print("importing datetime")
from datetime import datetime, timedelta
print("importing zipfile")
import zipfile
print("importing xml.etree.ElementTree")
import xml.etree.ElementTree as ET
print("importing pathlib")
from pathlib import Path
import shutil
end_time = time.time()
print(f"Time taken to import shutil: {end_time - start_time} seconds")
start_time = time.time()
import mimetypes
end_time = time.time()
print(f"Time taken to import mimetypes: {end_time - start_time} seconds")
start_time = time.time()
from PIL import Image, ImageDraw, ImageFont
end_time = time.time()
print(f"Time taken to import PIL: {end_time - start_time} seconds")
start_time = time.time()
print("importing Levenshtein")
import Levenshtein  # Add this import for Levenshtein distance
end_time = time.time()
print(f"Time taken to import Levenshtein: {end_time - start_time} seconds")
start_time = time.time()
print("importing Translator")
from googletrans import Translator  # Add this import for translation
end_time = time.time()
print(f"Time taken to import Translator: {end_time - start_time} seconds")
import uuid  # Add this import for UUID generation
import soundfile as sf
import numpy as np
import wave
from bs4 import BeautifulSoup  # Add this import at the top of the file
import ebooklib
from ebooklib import epub
import random
import re
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

LANG_ID = "fr"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
AUDIO_DIR = "frenchtrial.wav"

loaded_processors = {}
loaded_models = {}
translator = Translator()  

EPUB_DIR = Path("epub")
COVERS_DIR = Path("covers")
DEFAULT_COVER = "default-cover.png"

db = None

language_dict = {
    #"en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    #"es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    #"it": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    #"de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
}

def get_or_load_model(lang):
    """Helper function to get or load model and processor"""
    MODEL_ID = language_dict[lang]
    
    if lang not in loaded_processors:
        loaded_processors[lang] = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    if lang not in loaded_models:
        loaded_models[lang] = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
        
    return loaded_processors[lang], loaded_models[lang]

def stt(AUDIO_DIR, lang):
    start_time = time.time()
    processor, model = get_or_load_model(lang)   
    end_time = time.time()
    print(f"Time taken to load model: {end_time - start_time} seconds")
    
    start_time = time.time()
    try:
        # Convert audio to proper WAV format using ffmpeg if installed
        temp_wav = f"{AUDIO_DIR}_temp.wav"
        os.system(f"ffmpeg -y -i {AUDIO_DIR} -acodec pcm_s16le -ar 16000 -ac 1 {temp_wav}")
        
        with wave.open(temp_wav, 'rb') as wav_file:
            # Get audio data
            frames = wav_file.readframes(wav_file.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            
            # Normalize audio
            audio = audio / 32768.0
            
            # Get sample rate
            sr = wav_file.getframerate()
            
        # Clean up temporary file
        os.remove(temp_wav)
            
    except Exception as e:
        print(f"Error processing audio: {e}")
        return "Error processing audio file"

    end_time = time.time()
    print(f"Time taken to load audio: {end_time - start_time} seconds")
    
    start_time = time.time()
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    end_time = time.time()
    print(f"Time taken to load inputs: {end_time - start_time} seconds")
    
    start_time = time.time()
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    end_time = time.time()
    print(f"Time taken to load logits: {end_time - start_time} seconds")
    
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)
    
    print("Prediction:", predicted_sentences)
    return predicted_sentences[0]

class TextComparator:
    @staticmethod
    def generate_html_report(text1, text2, output_file='text_comparison_report.html'):
        overall_start = time.time()
        
        # Normalize text for comparison
        normalize_start = time.time()
        def normalize_text(text):
            text = text.lower()
            # Replace punctuation with spaces
            for char in ',.!?;:«»""()[]{}':
                text = text.replace(char, ' ')
            text = text.replace("'", "'")  # Standardize apostrophes
            return [word for word in text.split() if word]
        
        # Get normalized words
        original_words = normalize_text(text1)
        spoken_words = normalize_text(text2)
        normalize_end = time.time()
        print(f"Time for text normalization: {normalize_end - normalize_start:.4f} seconds")
        
        # Create a cache for Levenshtein distances
        distance_cache = {}
        def cached_levenshtein(word1, word2):
            key = (word1, word2)
            if key not in distance_cache:
                distance_cache[key] = Levenshtein.distance(word1, word2)
            return distance_cache[key]
        
        # Pre-calculate word similarities
        similarities_start = time.time()
        word_similarities = {}
        for i, orig in enumerate(original_words):
            # Only compare with nearby words
            start_idx = max(0, i - 3)
            end_idx = min(len(spoken_words), i + 4)
            for j in range(start_idx, end_idx):
                if j < len(spoken_words):
                    spok = spoken_words[j]
                    distance = cached_levenshtein(orig, spok)
                    max_len = max(len(orig), len(spok))
                    word_similarities[(orig, spok)] = 1 - (distance / max_len if max_len > 0 else 0)
        similarities_end = time.time()
        print(f"Time for word similarities calculation: {similarities_end - similarities_start:.4f} seconds")
        print(f"Number of Levenshtein calculations: {len(distance_cache)}")

        # Create matrices
        matrix_start = time.time()
        m, n = len(original_words), len(spoken_words)
        score = [[0 for _ in range(n+1)] for _ in range(m+1)]
        traceback = [[None for _ in range(n+1)] for _ in range(m+1)]
        
        # Initialize first row and column
        for i in range(m+1):
            score[i][0] = -i
            if i > 0: traceback[i][0] = "up"
        for j in range(n+1):
            score[0][j] = -j
            if j > 0: traceback[0][j] = "left"
        
        # Fill matrices
        for i in range(1, m+1):
            for j in range(1, n+1):
                word_similarity = word_similarities.get(
                    (original_words[i-1], spoken_words[j-1]),
                    0
                )
                
                match_score = score[i-1][j-1] + (2 * word_similarity - 1)
                delete_score = score[i-1][j] - 0.5
                insert_score = score[i][j-1] - 0.5
                
                best_score = max(match_score, delete_score, insert_score)
                score[i][j] = best_score
                
                if best_score == match_score:
                    traceback[i][j] = "diag"
                elif best_score == delete_score:
                    traceback[i][j] = "up"
                else:
                    traceback[i][j] = "left"
        matrix_end = time.time()
        print(f"Time for matrix operations: {matrix_end - matrix_start:.4f} seconds")

        # Traceback alignment
        alignment_start = time.time()
        aligned_original = []
        aligned_spoken = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and traceback[i][j] == "diag":
                aligned_original.append(original_words[i-1])
                aligned_spoken.append(spoken_words[j-1])
                i -= 1
                j -= 1
            elif i > 0 and traceback[i][j] == "up":
                aligned_original.append(original_words[i-1])
                aligned_spoken.append(None)
                i -= 1
            else:
                aligned_original.append(None)
                aligned_spoken.append(spoken_words[j-1])
                j -= 1
        
        aligned_original.reverse()
        aligned_spoken.reverse()
        alignment_end = time.time()
        print(f"Time for alignment: {alignment_end - alignment_start:.4f} seconds")
        
        # Generate HTML output
        html_start = time.time()
        marked_output = []
        for orig, spoken in zip(aligned_original, aligned_spoken):
            if orig is None:
                marked_output.append(f'<span id="" class="wrong" style="color:red;">{spoken}</span>')
            elif spoken is None:
                continue
            else:
                distance = cached_levenshtein(orig, spoken)
                max_len = max(len(orig), len(spoken))
                ratio = distance / max_len if max_len > 0 else 0
                
                if distance > 2 and ratio > 0.3:
                    marked_output.append(f'<span id="{orig}" class="wrong" style="color:red;">{spoken}</span>')
                else:
                    marked_output.append(spoken)
        
        marked_text = ' '.join(marked_output)
        similarity_ratio = difflib.SequenceMatcher(None, original_words, spoken_words).ratio()
        html_end = time.time()
        print(f"Time for HTML generation: {html_end - html_start:.4f} seconds")
        
        overall_end = time.time()
        print(f"Total processing time: {overall_end - overall_start:.4f} seconds")
        
        return marked_text, similarity_ratio, original_words, spoken_words

def stt_task(data_object):
    print(f"language: {data_object['language']}") 

    fpath=f"{data_object['username']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wav"
    with open(fpath, "wb") as audio_file:
            base64_string = data_object["blob"]
            actual_base64 = base64_string.split(',')[1]
            binary_data = base64.b64decode(actual_base64)
            audio_file.write(binary_data)
    the_words = stt(fpath,data_object["language"])
    
    print("\nStarting text comparison analysis...")
    predicted_sentence, similarity_ratio, original_words, spoken_words = TextComparator.generate_html_report(
        data_object["sentence"], 
        the_words
    )
    wrong_words = predicted_sentence.count('class="wrong"')
    # Simplified points system (temporarily bypassing Levenshtein)
    total_words = len(spoken_words)
    points = total_words  # Start with maximum points
    
  
    percentage_score = (points / total_words * 100) if total_words > 0 else 0
    
    print(f"\nPoints Summary:")
    print(f"Total words: {total_words}")
    print(f"Wrong words: {wrong_words}")
    print(f"Points earned: {points}/{total_words}")
    print(f"Percentage score: {percentage_score:.1f}%")
        
    message_returned = {
        "pred_sentence": predicted_sentence,
        "points": points,
        "total_words": total_words,
        "wrong_words": wrong_words,
        "percentage_score": round(percentage_score, 1)
    }
    
    db.set_current_book_task(data_object, fpath, predicted_sentence)

    if "current_book" in data_object and "username" in data_object:
        db.set_current_book_task({
            "username": data_object["username"],
            "book": data_object["current_book"],
            "page": data_object["page"]
        }, "", "")

    print("\nCompleted text comparison analysis")
    return message_returned

def extract_epub_cover(epub_path: Path, cover_dir: Path) -> str:
    """Extract cover image from EPUB file using the same approach as the PHP code"""
    try:
        with zipfile.ZipFile(epub_path) as zip_file:
            # First try META-INF/container.xml
            try:
                container = ET.fromstring(zip_file.read('META-INF/container.xml'))
                rootfile_path = container.find('.//{urn:oasis:names:tc:opendocol:xmlns:container}rootfile').get('full-path')
                
                # Read content.opf
                opf = ET.fromstring(zip_file.read(rootfile_path))
                
                # Look for cover image in manifest
                manifest = opf.find('.//{*}manifest')
                if manifest is not None:
                    # First try items with id containing 'cover'
                    for item in manifest.findall('.//{*}item'):
                        if 'cover' in item.get('id', '').lower():
                            href = item.get('href')
                            if href:
                                # Handle relative paths
                                if not href.startswith('/'):
                                    opf_dir = Path(rootfile_path).parent
                                    href = str(opf_dir / href)
                                
                                # Extract and save the cover
                                try:
                                    image_data = zip_file.read(href.lstrip('/'))
                                    cover_dir.mkdir(parents=True, exist_ok=True)
                                    output_path = cover_dir / f"{epub_path.stem}.jpg"
                                    output_path.write_bytes(image_data)
                                    return str(output_path.relative_to(COVERS_DIR.parent))
                                except Exception as e:
                                    print(f"Error saving cover image: {e}")
                                    continue
                    
                    # If no cover found, try first image in manifest
                    for item in manifest.findall('.//{*}item'):
                        media_type = item.get('media-type', '')
                        if media_type.startswith('image/'):
                            href = item.get('href')
                            if href:
                                # Handle relative paths
                                if not href.startswith('/'):
                                    opf_dir = Path(rootfile_path).parent
                                    href = str(opf_dir / href)
                                
                                # Extract and save the cover
                                try:
                                    image_data = zip_file.read(href.lstrip('/'))
                                    cover_dir.mkdir(parents=True, exist_ok=True)
                                    output_path = cover_dir / f"{epub_path.stem}.jpg"
                                    output_path.write_bytes(image_data)
                                    return str(output_path.relative_to(COVERS_DIR.parent))
                                except Exception as e:
                                    print(f"Error saving first image as cover: {e}")
                                    continue
            
            except Exception as e:
                print(f"Error reading EPUB structure: {e}")
                # Try direct image search as fallback
                for filename in zip_file.namelist():
                    if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                        try:
                            image_data = zip_file.read(filename)
                            cover_dir.mkdir(parents=True, exist_ok=True)
                            output_path = cover_dir / f"{epub_path.stem}.jpg"
                            output_path.write_bytes(image_data)
                            return str(output_path.relative_to(COVERS_DIR.parent))
                        except Exception as e:
                            print(f"Error saving fallback image: {e}")
                            continue
    
    except Exception as e:
        print(f"Error extracting cover from {epub_path}: {e}")
    
    print(f"No cover found for {epub_path.name}, using default")
    return DEFAULT_COVER

def get_file_as_base64(file_path: Path) -> str:
    """Convert a file to base64 string with proper mime type prefix"""
    try:
        mime_type = mimetypes.guess_type(str(file_path))[0]
        with open(file_path, 'rb') as file:
            b64_data = base64.b64encode(file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{b64_data}"
    except Exception as e:
        print(f"Error converting file to base64: {e}")
        return ""

async def get_available_books():
    """Get list of available EPUB books and their covers"""
    books = []
    
    try:
        # Make sure default cover exists
        default_cover_path = Path("images/default-cover.png")
        if not default_cover_path.exists():
            print(f"Creating default cover at {default_cover_path}")
            img = Image.new('RGB', (120, 180), color='lightgray')
            d = ImageDraw.Draw(img)
            d.text((10,10), "No\nCover", fill='black')
            default_cover_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(default_cover_path)
            print("Default cover created successfully")
        
        # Process books
        for lang_dir in EPUB_DIR.glob("*"):
            if lang_dir.is_dir():
                language = lang_dir.name
                print(f"\nProcessing language directory: {language}")
                
                for epub_path in lang_dir.glob("*.epub"):
                    print(f"\nProcessing book: {epub_path.name}")
                    cover_dir = COVERS_DIR / language
                    cover_path = extract_epub_cover(epub_path, cover_dir)
                    
                    try:
                        if cover_path == DEFAULT_COVER:
                            cover_file = default_cover_path
                        else:
                            cover_file = COVERS_DIR.parent / cover_path
                        
                        print(f"Converting cover to base64 for {epub_path.name}")
                        cover_base64 = get_file_as_base64(cover_file)
                        
                        books.append({
                            "filename": epub_path.name,
                            "language": language,
                            "path": str(epub_path.relative_to(EPUB_DIR.parent)),
                            "cover": cover_base64
                            # Removed the epub base64 data to make the response lighter
                        })
                        print(f"Successfully processed book: {epub_path.name}")
                    except Exception as e:
                        print(f"Error processing book {epub_path.name}: {e}")
                        continue
        
        return books
    except Exception as e:
        print(f"Error getting available books: {e}")
        return []

async def get_book_data(data_object):
    """Get specific book data by filename and language"""
    try:
        filename = data_object.get("filename")
        
        if not filename :
            return {"status": "error", "message": "Missing filename or language"}
        
        epubs = os.walk(EPUB_DIR)
        for root, dirs, files in epubs:
            for file in files:
                if file == filename:
                    epub_path = Path(root) / file
                    break
        print(epub_path)
        language = epub_path.parent.name
        print(f"language: {language}")
        if not epub_path.exists():
            return {"status": "error", "message": f"Book not found: {filename}"}
        
        # Convert the EPUB file to base64
        epub_base64 = get_file_as_base64(epub_path)
        
        return {
            "status": "success",
            "filename": filename,
            "language": language,
            "epub": epub_base64
        }
    except Exception as e:
        print(f"Error getting book data: {e}")
        return {"status": "error", "message": str(e)}

async def translate_task(data_object):
    source_text = data_object.get("text", "")
    source_lang = data_object.get("source_lang", "en")
    target_lang = data_object.get("target_lang", "fr")
    current_book = data_object.get("current_book", "")
    cfi = data_object.get("cfi", 0)
    username = data_object.get("username", "")  # Add username from data_object
    
    if not source_text:
        return {"status": "error", "message": "No text provided for translation"}
    
    if source_lang == target_lang:
        return {"status": "success", "translated_text": source_text}
    
    # Map language codes to standard codes if needed
    source_lang_map = {
        "francais": "fr",
        "english": "en",
        "español": "es",
        "espagnol": "es",
        "spanish": "es",
        "deutsch": "de",
        "italiano": "it"
    }
    
    # Convert language names to standard codes
    if source_lang in source_lang_map:
        source_lang = source_lang_map[source_lang]
    
    print(f"Attempting to translate from {source_lang} to {target_lang}")
    
    # Create a new translator instance for each request
    translator_instance = Translator()
    result = await translator_instance.translate(source_text, src=source_lang, dest=target_lang) #await on linux
    
    if hasattr(result, 'text'):
        translated_text = result.text
        print(f"Translation successful: {translated_text}")
        print(f"current_book: {current_book}")
        print(f"username: {username}")
        if current_book and username:  
            print(f"Updating database for user {username} with book {current_book} and cfi {cfi}")
            db.set_current_book_task({
                "username": username,
                "book": current_book,
                "cfi": cfi
            }, "", "")
        
        return {
            "status": "success", 
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "current_book": current_book,  # Add these fields to the response
        }
    else:
        print(f"Translation result has no 'text' attribute: {result}")
        return {
            "status": "error",
            "message": "Translation result format unexpected",
            "source_lang": source_lang,
            "target_lang": target_lang
        }
   

async def login_task(self, data_object):
    # Initialize message and variables at the start
    message = {"status": "error", "message": "Invalid credentials"}
    epub = None
    language = None
    
    conn = sqlite3.connect(self.DB_PATH)
    cursor = conn.cursor()
    
    username = data_object.get("username")
    password = data_object.get("password")
    
    # Get column names
    cursor.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in cursor.fetchall()]
    
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
                (username, password))
    user_row = cursor.fetchone()
    
    if user_row:
        # Convert row to dictionary
        user = dict(zip(columns, user_row))
        print(user)
        
        # Generate new token using UUID and set expiry to 24 hours from now
        token = str(uuid.uuid4())
        expiry = datetime.now() + timedelta(days=10000)
        
        cursor.execute("""
            UPDATE users 
            SET login_token = ?, token_expiry = ? 
            WHERE username = ?""", 
            (token, expiry, username))
        conn.commit()

        print(f"user.get('current_book'): {user.get('current_book')}")    
        if user.get("current_book"):
            print("getting books")
            epubs = os.walk(EPUB_DIR)
            for root, dirs, files in epubs:
                for file in files:
                    if file == user["current_book"]:
                        book_path = Path(root) / file
                        break
            print(f"book_path: {book_path}")
            if 'book_path' in locals() and book_path.exists():
                language = book_path.parent.name
                print(f"language: {language}")
                with open(book_path, "rb") as f:
                    book_data = f.read()
                    book_base64 = base64.b64encode(book_data).decode('utf-8')
                    epub = f"data:application/epub+zip;base64,{book_base64}"
                    print(f"Including book data for {user['current_book']}")

        message = {
            "status": "success",
            "token": token,
            "username": username,
            "current_book": user.get("current_book", ""),
            "book_position": user.get("book_position", 0),
            "preferred_language": user.get("preferred_language", "en"),
            "type": "login"
        }

        # Only add epub and language to message if they were set
        if epub is not None:
            message["epub"] = epub
        if language is not None:
            message["language"] = language
    
    conn.close()
    return message

async def verify_token_task(data_object):
    """Verify a login token and return user data if valid"""
    conn = sqlite3.connect(db.DB_PATH)
    cursor = conn.cursor()
    
    token = data_object.get("token")
    if not token:
        return {"status": "error", "message": "No token provided"}
    
    # Get column names
    cursor.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in cursor.fetchall()]
    print(columns)
    
    # Check if token exists and is not expired
    cursor.execute("""
        SELECT * FROM users 
        WHERE login_token = ? AND token_expiry > ?
    """, (token, datetime.now()))
    
    user_row = cursor.fetchall()
    conn.close()
    
    if not user_row:
        return {"status": "error", "message": "Invalid or expired token"}
    
    user_row = user_row[0]
    # Convert row to dictionary
    user = dict(zip(columns, user_row))
    print("User data:", user)
    
    

    print(f"user.get('current_book'): {user.get('current_book')}")    
    if user.get("current_book"):
        print("getting books")
        epubs = os.walk(EPUB_DIR)
        for root, dirs, files in epubs:
            for file in files:
                if file == user["current_book"]:
                    book_path = Path(root) / file
                    break
        print(f"book_path: {book_path}")
        if book_path.exists():
            language = book_path.parent.name
            print(f"language: {language}")
            with open(book_path, "rb") as f:
                book_data = f.read()
                book_base64 = base64.b64encode(book_data).decode('utf-8')
                epub = f"data:application/epub+zip;base64,{book_base64}"
                print(f"Including book data for {user['current_book']}")
    else:
        epub = None
        language = None
    response = {
        "status": "success",
        "username": user.get("username", ""),
        "current_book": user.get("current_book", ""),
        "preferred_language": user.get("preferred_language", "en"),
        "token": token,
        "epub": epub,
        "language": language,
        "cfi": user.get("cfi", "")
    }
    return response


async def handle_connection(websocket):
    print(f"Client connected, {websocket.remote_address}")
    
    try:
        data = await websocket.recv()
        data_object = json.loads(data)
        print(f"Received data: {data_object}")
        message_returned = {"error": "Invalid task"}  # Default message
        
        if data_object.get("task") == "get_books":
            # Handle book listing request
            books = await get_available_books()
            message_returned = {"books": books}
        elif data_object.get("task") == "get_book_data":
            message_returned = await get_book_data(data_object)
        elif data_object.get("task") == "stt":
            message_returned = stt_task(data_object)
        elif data_object.get("task") == "login":
            message_returned = await db.login_task(data_object)
        elif data_object.get("task") == "signup":
            message_returned = db.signup_task(data_object)
        elif data_object.get("task") == "change_settings":
            message_returned = db.change_settings_task(data_object)
        elif data_object.get("task") == "translate":
            message_returned = await translate_task(data_object)
        elif data_object.get("task") == "verify_token":
            message_returned = await verify_token_task(data_object)
        elif data_object.get("task") == "pagele":
            message_returned = pagele_task(data_object)  # Remove await since it's not async
            
        await websocket.send(json.dumps(message_returned))
    
    except Exception as e:
        print(f"Error: {e}")
        await websocket.send(json.dumps({"error": str(e)}))
    finally:
        print("Client disconnected")

# Add a function to preload all models
def preload_all_models():
    """Preload all language models at startup"""
    print("Preloading all language models...")
    for lang in language_dict:
        print(f"Loading {lang} model...")
        get_or_load_model(lang)
    print("All models loaded!")

class DatabaseManager:
    def __init__(self, db_path):
        self.DB_PATH = db_path #a constructor defines what parameters are needed to create an object

    def init_database(self):
        """Initialize the SQLite database with necessary tables"""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        # Create users table with CFI instead of page
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            current_book TEXT DEFAULT '',
            cfi TEXT DEFAULT '',  -- Changed from page to cfi
            utterrance_fname TEXT DEFAULT '',
            predicted_sentence TEXT DEFAULT '',
            preferred_language TEXT DEFAULT 'en',
            login_token TEXT,
            token_expiry TIMESTAMP
        )
        ''')

        conn.commit()
        conn.close()

    def signup_task(self, data_object):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        username = data_object.get("username")
        password = data_object.get("password")
        
        # Check if username already exists
        cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            return {"status": "error", "message": "Username already exists"}
        
        # Generate token for new user
        token = str(uuid.uuid4())
        expiry = datetime.now() + timedelta(days=10000)
        
        # Create new user with token
        cursor.execute("""
            INSERT INTO users (username, password, login_token, token_expiry) 
            VALUES (?, ?, ?, ?)""", 
            (username, password, token, expiry))
        conn.commit()
        conn.close()
        
        return {
            "status": "success", 
            "message": "User created successfully",
            "token": token,
            "username": username
        }

    def set_current_book_task(self, data_object, utterrance_fname, predicted_sentence):
        try:
            conn = sqlite3.connect(self.DB_PATH)
            cursor = conn.cursor()
            
            username = data_object.get("username")
            current_book = data_object.get("book", data_object.get("current_book", ""))
            
            # Store the full CFI string instead of trying to convert it
            cfi = data_object.get("cfi", data_object.get("page", ""))
            if isinstance(cfi, (int, float)):  # If it's a number, convert to string
                cfi = str(cfi)

            print(f"Received CFI from client: {cfi}")
            print(f"Updating database for {username} with book: {current_book}, cfi: {cfi}")
            
            # First verify the user exists
            cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
            if not cursor.fetchone():
                print(f"User {username} not found in database")
                return False

            # Update the user's current book and cfi
            cursor.execute("""
                UPDATE users 
                SET current_book = ?, 
                    cfi = ?, 
                    utterrance_fname = ?, 
                    predicted_sentence = ? 
                WHERE username = ?
            """, (current_book, cfi, utterrance_fname, predicted_sentence, username))
            
            if cursor.rowcount == 0:
                print(f"No rows were updated for user {username}")
            else:
                print(f"Successfully updated book info for user {username}")
            
            conn.commit()
            
            # Verify the update
            cursor.execute("SELECT current_book, cfi FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            print(f"After update - Book: {result[0]}, CFI: {result[1]}")
            
            return True
        except Exception as e:
            print(f"Error in set_current_book_task: {e}")
            return False
        finally:
            conn.close()

    def change_settings_task(self, data_object):
        """Handle user settings updates"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        token = data_object.get("token")
        updates = {}
        
        if "current_book" in data_object:
            updates["current_book"] = data_object["current_book"]
        if "book_position" in data_object:
            updates["book_position"] = data_object["book_position"]
        if "preferred_language" in data_object:
            updates["preferred_language"] = data_object["preferred_language"]
        
        if not updates:
            return {"status": "error", "message": "No settings to update"}
        
        # Verify token and update settings
        cursor.execute("SELECT username FROM users WHERE login_token = ? AND token_expiry > ?", 
                    (token, datetime.now()))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return {"status": "error", "message": "Invalid or expired token"}
        
        # Build update query
        update_query = "UPDATE users SET " + ", ".join(f"{k} = ?" for k in updates.keys())
        update_query += " WHERE login_token = ?"
        
        # Execute update
        cursor.execute(update_query, list(updates.values()) + [token])
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "Settings updated successfully"}


async def main():

    global db 
    db = DatabaseManager("usersHablas.db")
    db.init_database();
    preload_all_models()
    print("Preloaded all models")

    async with websockets.serve(handle_connection, "localhost", 8675) as server:
        print("WebSocket server started on local NOT wss://carriertech.uk:8675")
        await asyncio.Future()  # Run forever 

if __name__ == "__main__": #when you use a multi processer load, you use this so it doesnt crash. with asyncio you always have to do it
    print("we're initializing where tf is the error??")
    asyncio.run(main()) 



