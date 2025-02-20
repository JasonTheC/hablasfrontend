import asyncio, difflib #multiprocessser good for sockets
import websockets , json # the json is bc we're sending a json object from the website
import wave , base64
import os #library for anything that has to do with your hard-drive
import torch, sqlite3
import librosa #library to analyse and process audio . soundfile is similar 
import os #library for anything that has to do with your hard-drive
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import ssl  # Add this import at the top
import secrets  # Add secrets to generate tokens
from datetime import datetime, timedelta



LANG_ID = "fr"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
AUDIO_DIR = "frenchtrial.wav"

# Add these global variables at the top level after the imports
loaded_processors = {}
loaded_models = {}

# Add these database constants after other constants
DB_PATH = "users.db"

language_dict = {
    "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
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
    # Get or load the model and processor
    processor, model = get_or_load_model(lang)
    
    # Pre-processing the data
    audio = librosa.load(AUDIO_DIR, sr=16_000)
    print(type(audio[0]))

    # Tokenizing the data
    inputs = processor(audio[0], sampling_rate=audio[1], return_tensors="pt")

    # Inference process
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)
    
    print("Prediction:", predicted_sentences)
    return predicted_sentences[0]

class TextComparator:
    @staticmethod
    def generate_html_report(text1, text2, output_file='text_comparison_report.html'):
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        matcher = difflib.SequenceMatcher(None, words1, words2)       
        marked_words2 = words2.copy()        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            print((tag,i1, i2, j1, j2))
            if tag != 'equal':
                for j in range(j1, j2):
                    # Set id to the corresponding word in words1 if it exists, otherwise use empty string
                    corresponding_word = words1[i1] if i1 < len(words1) else ""
                    marked_words2[j] = f'<span id="{corresponding_word}" class="wrong" style="color:red;">{marked_words2[j]}</span>'
        
        marked_text2 = ' '.join(marked_words2)
        
        similarity_ratio = matcher.ratio()
        
       
        return marked_text2, similarity_ratio

def stt_task(data_object):
    print(f"language: {data_object['language']}") 

    fpath="received_audio.wav" #name of the audio in the server (our computer in this case)
    with open(fpath, "wb") as audio_file: #received_audio is f.open, initialising this file. audio_file is the variable
            base64_string = data_object["blob"]
            actual_base64 = base64_string.split(',')[1]  # Get the Base64 part
            binary_data = base64.b64decode(actual_base64)
            audio_file.write(binary_data)
    the_words = stt(fpath,data_object["language"])
    predicted_sentence, similarity_ratio = TextComparator.generate_html_report(data_object["sentence"], the_words)
    message_returned = {"pred_sentence":predicted_sentence}

    print("Received audio file and saved as 'received_audio.wav'")
    return message_returned

async def handle_connection(websocket):
    print(f"Client connected, {websocket.remote_address}")

    
    try:
        data= await websocket.recv()
        data_object = json.loads(data)

        """
        data_object = {
            "language": "fr",
            "sentence": "Bonjour, comment ça va ?",
            "blob": "base64_encoded_audio_data",
            "task": "stt"
        }
        """
        if data_object["task"] == "stt":
            message_returned = stt_task(data_object)
        elif data_object["task"] == "login":
            message_returned = login_task(data_object)
        elif data_object["task"] == "signup":
            message_returned = signup_task(data_object)
        elif data_object["task"] == "change_settings":
            message_returned = change_settings_task(data_object)
        else:
            message_returned = {"error": "Invalid task"}
        
        await websocket.send(json.dumps(message_returned)) #waits until the sockets has sent everything and the pipe closes
    
    except Exception as e:
        print(f"Error: {e}")
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

def init_database():
    """Initialize the SQLite database with necessary tables"""
    conn = sqlite3.connect(DB_PATH)
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
    
    conn.commit()
    conn.close()

def login_task(data_object):
    """Handle user login"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    username = data_object.get("username")
    password = data_object.get("password")
    
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
                  (username, password))
    user = cursor.fetchone()
    
    if user:
        # Generate new token and set expiry to 24 hours from now
        token = secrets.token_urlsafe(32)
        expiry = datetime.now() + timedelta(days=1)
        
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
            "book_position": user[3],
            "preferred_language": user[4]
        }
    else:
        message = {
            "status": "error",
            "message": "Invalid credentials"
        }
    
    conn.close()
    return message

def signup_task(data_object):
    """Handle user registration"""
    conn = sqlite3.connect(DB_PATH)
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

def change_settings_task(data_object):
    """Handle user settings updates"""
    conn = sqlite3.connect(DB_PATH)
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

# Modify the main function to preload models at startup
async def main():
    # Initialize database
    if not os.path.isfile(DB_PATH): init_database()
    
    # Preload all models before starting the server
    preload_all_models()
    
    # Create SSL context
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(
        '/media/nas/SSLCerts/carriertech.uk/fullchain.pem',
        '/media/nas/SSLCerts/carriertech.uk/privkey.pem'
    )

    # Start the WebSocket server with SSL
    async with websockets.serve(
        handle_connection, 
        "0.0.0.0",  # Changed from localhost to accept external connections
        8675, 
        ssl=ssl_context
    ):
        print("WebSocket server started on wss://carriertech.uk:8675")
        await asyncio.Future()  # Run forever


if __name__ == "__main__": #when you use a multi processer load, you use this so it doesnt crash. with asyncio you always have to do it
    asyncio.run(main())
