import json
import os
from deep_translator import GoogleTranslator
from typing import Dict, Any
import re
import time

def create_translation_structure(text: str, translator: GoogleTranslator) -> Dict[str, Any]:
    # Remove punctuation from text
    text_no_punct = text.translate(str.maketrans('', '', string.punctuation))
    
    # Split text into words
    words = text_no_punct.split()
    
    full_translation = translator.translate(text)
    
    # Create word-level mapping
    word_mapping = {}
    for idx, word in enumerate(words, 1):
        print(f"word: {word}")
        if word.strip():  # Only translate non-empty words
            translated_word = translator.translate(word)
            print(f"translated_word: {translated_word}")
            word_mapping[str(idx)] = translated_word
            # Add a small delay to avoid rate limiting
            time.sleep(0.1)
        else:
            input(f"no word?")
           
    print(word_mapping)
    return {
        "translation": full_translation,
        "words": word_mapping
    }

def process_json_file(input_file: str, output_file: str, lang_dir, target_lang: str):
    print(f"{lang_dir[:2]} to {target_lang[:2]}")
    translator = GoogleTranslator(source=lang_dir[:2], target=target_lang[:2] )   
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each chapter
    translated_data = {}
    for chapter, content in data.items():
        print(f"Processing chapter: {chapter}")
        # Initialize list for this chapter's translations
        translated_data[chapter] = []
        
        # Process each string in the chapter's content list
        for i, text in enumerate(content):
            print(f"  Processing string {i+1}/{len(content)}")
            translation = create_translation_structure(text, translator)
            translated_data[chapter].append(translation)
            time.sleep(0.5)
       
    
    # Save translated data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

def main():
    # Define language codes mapping
    lang_codes = {
        'english': 'en',
        #'espagnol': 'es',
        #'francais': 'fr',
        #'deutsch': 'de',
        #'italiano': 'it',
        #'portuguese': 'pt',
        #'turkish': 'tr'
    }
    
    # Process each language directory
    for lang_dir in os.listdir('epub'):
        for lang_target in lang_codes.keys():
            if lang_dir == lang_codes:continue
            input_dir = os.path.join('epub', lang_dir)
            output_dir = os.path.join('epub_translations', lang_dir)
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Process each JSON file in the language directory
            for json_file in os.listdir(input_dir):
                if json_file.endswith('.json'):
                    input_file = os.path.join(input_dir, json_file)
                    output_file = os.path.join(output_dir, json_file)
                    
                    print(f"Processing {input_file}...")
                    process_json_file(input_file, output_file, lang_dir, lang_target)
                    print(f"Created translation at {output_file}")

if __name__ == "__main__":
    main() 