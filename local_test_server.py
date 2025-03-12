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
import secrets  # Add secrets to generate tokens
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


LANG_ID = "fr"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
AUDIO_DIR = "frenchtrial.wav"

# Add these global variables at the top level after the imports
loaded_processors = {}
loaded_models = {}
translator = Translator()  # Initialize the translator

# Add these database constants after other constants
DB_PATH = "users.db"

# Add these constants after other constants
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
    processor, model = get_or_load_model(lang)    
    audio = librosa.load(AUDIO_DIR, sr=16_000)
    inputs = processor(audio[0], sampling_rate=audio[1], return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)
    
    print("Prediction:", predicted_sentences)
    return predicted_sentences[0]

class TextComparator:
    @staticmethod
    def generate_html_report(text1, text2, output_file='text_comparison_report.html'):
        # Normalize text for comparison
        def normalize_text(text):
            text = text.lower()
            # Replace punctuation with spaces
            for char in ',.!?;:«»""()[]{}':
                text = text.replace(char, ' ')
            
            # Handle apostrophes specially
            text = text.replace("'", "'")  # Standardize apostrophes
            
            # Split into words and filter empty strings
            return [word for word in text.split() if word]
        
        # Get normalized words
        original_words = normalize_text(text1)
        spoken_words = normalize_text(text2)
        
        print(f"Original words: {original_words}")
        print(f"Spoken words: {spoken_words}")
        
        # Create a dynamic programming matrix for alignment
        # This is similar to the Needleman-Wunsch algorithm for sequence alignment
        m, n = len(original_words), len(spoken_words)
        
        # Initialize the score matrix
        score = [[0 for _ in range(n+1)] for _ in range(m+1)]
        
        # Initialize the traceback matrix
        traceback = [[None for _ in range(n+1)] for _ in range(m+1)]
        
        # Fill the first row and column with gap penalties
        for i in range(m+1):
            score[i][0] = -i
            if i > 0:
                traceback[i][0] = "up"
        
        for j in range(n+1):
            score[0][j] = -j
            if j > 0:
                traceback[0][j] = "left"
        
        # Fill the score and traceback matrices
        for i in range(1, m+1):
            for j in range(1, n+1):
                # Calculate similarity score between words
                word_similarity = 1 - Levenshtein.distance(original_words[i-1], spoken_words[j-1]) / max(len(original_words[i-1]), len(spoken_words[j-1]))
                
                # Calculate scores for different moves
                match_score = score[i-1][j-1] + (2 * word_similarity - 1)  # Reward for similar words, penalty for different
                delete_score = score[i-1][j] - 0.5  # Gap penalty
                insert_score = score[i][j-1] - 0.5  # Gap penalty
                
                # Choose the best move
                best_score = max(match_score, delete_score, insert_score)
                score[i][j] = best_score
                
                # Record the move in the traceback matrix
                if best_score == match_score:
                    traceback[i][j] = "diag"
                elif best_score == delete_score:
                    traceback[i][j] = "up"
                else:
                    traceback[i][j] = "left"
        
        # Traceback to find the alignment
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
                aligned_spoken.append(None)  # Gap in spoken
                i -= 1
            else:  # traceback[i][j] == "left"
                aligned_original.append(None)  # Gap in original
                aligned_spoken.append(spoken_words[j-1])
                j -= 1
        
        # Reverse the alignments
        aligned_original.reverse()
        aligned_spoken.reverse()
        
        # Generate HTML output
        marked_output = []
        
        for orig, spoken in zip(aligned_original, aligned_spoken):
            if orig is None:
                # Extra word in spoken text
                marked_output.append(f'<span id="" class="wrong" style="color:red;">{spoken}</span>')
            elif spoken is None:
                # Missing word in spoken text (optional to include)
                continue
            else:
                # Both words exist - check similarity
                distance = Levenshtein.distance(orig, spoken)
                max_len = max(len(orig), len(spoken))
                ratio = distance / max_len if max_len > 0 else 0
                
                if distance > 2 and ratio > 0.3:
                    marked_output.append(f'<span id="{orig}" class="wrong" style="color:red;">{spoken}</span>')
                else:
                    marked_output.append(spoken)
        
        # Join the marked words back into text
        marked_text = ' '.join(marked_output)
        
        # Calculate overall similarity
        similarity_ratio = difflib.SequenceMatcher(None, original_words, spoken_words).ratio()
        
        return marked_text, similarity_ratio, original_words, spoken_words

def stt_task(data_object):
    print(f"language: {data_object['language']}") 

    fpath=f"{data_object['username']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wav" #name of the audio in the server (our computer in this case)
    with open(fpath, "wb") as audio_file: #received_audio is f.open, initialising this file. audio_file is the variable
            base64_string = data_object["blob"]
            actual_base64 = base64_string.split(',')[1]  # Get the Base64 part
            binary_data = base64.b64decode(actual_base64)
            audio_file.write(binary_data)
    the_words = stt(fpath,data_object["language"])
    predicted_sentence, similarity_ratio, original_words, spoken_words = TextComparator.generate_html_report(data_object["sentence"], the_words)
    # Print detailed debug information
    for i in range(min(len(original_words), len(spoken_words))):
        distance = Levenshtein.distance(original_words[i], spoken_words[i])
        max_len = max(len(original_words[i]), len(spoken_words[i]))
        ratio = distance / max_len if max_len > 0 else 0
        print(f"Word {i+1}: Original '{original_words[i]}' vs Spoken '{spoken_words[i]}'")
        print(f"  - Levenshtein distance: {distance}")
        print(f"  - Max length: {max_len}")
        print(f"  - Distance ratio: {ratio:.2f}")
        print(f"  - Marked as wrong: {distance > 2 and ratio > 0.3}")
    message_returned = {"pred_sentence":predicted_sentence}
    db.set_current_book_task(data_object, fpath, predicted_sentence)
    print("Received audio file and saved as 'received_audio.wav'")
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
        language = data_object.get("language")
        
        if not filename or not language:
            return {"status": "error", "message": "Missing filename or language"}
        
        # Construct the full path to the book
        epub_path = EPUB_DIR / language / filename
        
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
    
    if not source_text:
        return {"status": "error", "message": "No text provided for translation"}
    
    if source_lang == target_lang:
        return {"status": "success", "translated_text": source_text}
    
    # Map language codes to standard codes if needed
    source_lang_map = {
        "francais": "fr",
        "english": "en",
        "español": "es",
        "deutsch": "de",
        "italiano": "it"
    }
    
    # Convert language names to standard codes
    if source_lang in source_lang_map:
        source_lang = source_lang_map[source_lang]
    
    try:
        # Use a simpler approach with direct translation
        print(f"Attempting to translate from {source_lang} to {target_lang}")
        
        # Create a new translator instance for each request
        translator_instance = Translator()
        result = translator_instance.translate(source_text, src=source_lang, dest=target_lang)
        
        if hasattr(result, 'text'):
            translated_text = result.text
            print(f"Translation successful: {translated_text}")
            return {
                "status": "success", 
                "translated_text": translated_text,
                "source_lang": source_lang,
                "target_lang": target_lang
            }
        else:
            print(f"Translation result has no 'text' attribute: {result}")
            return {
                "status": "error",
                "message": "Translation result format unexpected",
                "source_lang": source_lang,
                "target_lang": target_lang
            }
    except Exception as e:
        print(f"Translation error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "source_lang": source_lang,
            "target_lang": target_lang
        }

async def verify_token_task(data_object):
    """Verify a login token and return user data if valid"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    token = data_object.get("token")
    if not token:
        return {"status": "error", "message": "No token provided"}
    
    # Check if token exists and is not expired
    cursor.execute("""
        SELECT username, current_book, page, utterrance_fname, predicted_sentence, 
               preferred_language
        FROM users 
        WHERE login_token = ? AND token_expiry > ?
    """, (token, datetime.now()))
    
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return {"status": "error", "message": "Invalid or expired token"}
    
    # Return user data
    return {
        "status": "success",
        "username": user[0],
        "current_book": user[1],
        "page": user[2],
        "utterance_fname": user[3],
        "predicted_sentence": user[4],
        "preferred_language": user[5],
        "token": token  # Return the same token back
    }

async def handle_connection(websocket):
    print(f"Client connected, {websocket.remote_address}")
    
    try:
        data = await websocket.recv()
        data_object = json.loads(data)
        
        message_returned = {"error": "Invalid task"}  # Default message
        
        if data_object.get("task") == "get_books":
            # Handle book listing request
            books = await get_available_books()
            message_returned = {"books": books}
        elif data_object.get("task") == "get_book_data":
            # Handle specific book data request
            message_returned = await get_book_data(data_object)
        elif data_object.get("task") == "stt":
            message_returned = stt_task(data_object)
        elif data_object.get("task") == "login":
            message_returned = db.login_task(data_object)
        elif data_object.get("task") == "signup":
            message_returned = db.signup_task(data_object)
        elif data_object.get("task") == "change_settings":
            message_returned = db.change_settings_task(data_object)
        elif data_object.get("task") == "translate":
            message_returned = await translate_task(data_object)
        elif data_object.get("task") == "verify_token":
            message_returned = await verify_token_task(data_object)
            
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
        
        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users ( 
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            current_book TEXT DEFAULT '',
            book_position INTEGER DEFAULT 0,
            preferred_language TEXT DEFAULT 'en',
            login_token TEXT,
            token_expiry TIMESTAMP
        )
        ''')
        #the table is named users
        conn.commit()
        conn.close()

    async def login_task(self, data_object):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        username = data_object.get("username")
        password = data_object.get("password")
        
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
                    (username, password))
        user = cursor.fetchone()
        
        if user:
            print(user)
            # Generate new token and set expiry to 24 hours from now
            token = secrets.token_urlsafe(32)
            expiry = datetime.now() + timedelta(days=10000)
            
            cursor.execute("""
                UPDATE users 
                SET login_token = ?, token_expiry = ? 
                WHERE username = ?""", 
                (token, expiry, username))
            conn.commit()
            
            message = {
                "status": "success",
                "token": token,
                "username": username,
                "current_book": user[2],
                "page": user[3],
                "utterance_fname": user[4],
                "predicted_sentence": user[5],
                "book_position": user[3],
                "preferred_language": user[4],
                "type": "login"
            }
        else:
            message = {
                "status": "error",
                "message": "Invalid credentials"
            }
        
        conn.close()
        return message
        

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
        
        # Create new user
        cursor.execute("""
            INSERT INTO users (username, password) 
            VALUES (?, ?)""", 
            (username, password))
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "User created successfully"}

    def set_current_book_task(self, data_object, utterrance_fname, predicted_sentence):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        username = data_object.get("username")
        current_book = data_object.get("book")
        page = data_object.get("page")

        cursor.execute("UPDATE users SET current_book = ?, page = ?, utterrance_fname = ?, predicted_sentence = ? WHERE username = ?",
                    (current_book, page, utterrance_fname, predicted_sentence, username))
        conn.commit()
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

    async with websockets.serve(handle_connection, "localhost", 8765) as server:
        print("WebSocket server started on local NOT wss://carriertech.uk:8765")
        await asyncio.Future()  # Run forever 

if __name__ == "__main__": #when you use a multi processer load, you use this so it doesnt crash. with asyncio you always have to do it
    print("we're initializing where tf is the error??")
    asyncio.run(main()) 



