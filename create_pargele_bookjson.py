import ebooklib
from ebooklib import epub
import json
import re
from bs4 import BeautifulSoup
import nltk
import os
# Download both punkt and the French models
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab_french')
from nltk.tokenize import sent_tokenize

# File path
bdir = "./epub/espagnol/Plateroyyo.epub"

# Create debug folder
debug_dir = "./debug_extraction"
os.makedirs(debug_dir, exist_ok=True)

# Function to clean HTML content
def clean_html_content(content):
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')
    # Get text content
    text = soup.get_text()
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_main_chapter_text(text):
    # More lenient chapter detection
    return bool(re.match(r'.*?[IVX]+[\s\.]+', text[:50]))

def extract_main_chapter(text, item_id):
    # Save the raw text for debugging
    with open(f"{debug_dir}/{item_id}_raw.txt", 'w', encoding='utf-8') as f:
        f.write(text)
    
    # Try to identify chapter pattern
    chapter_match = re.search(r'([IVX]+[\s\.]+.*?)(?:CUESTIONARIO|REPASO|MODISMOS|TRADUCCIÓN|REDACCIÓN|$)', 
                              text, re.DOTALL)
    
    if chapter_match:
        main_text = chapter_match.group(1).strip()
        # Clean footnote references
        main_text = re.sub(r'\[Footnote \d+:.*?\]', '', main_text)
        main_text = re.sub(r'\[\d+\]', '', main_text)
        
        # Save the extracted chapter for debugging
        with open(f"{debug_dir}/{item_id}_chapter.txt", 'w', encoding='utf-8') as f:
            f.write(main_text)
            
        return main_text
    
    return None

# Load the EPUB book
try:
    print(f"Loading EPUB from: {bdir}")
    book = epub.read_epub(bdir)
    print(f"EPUB loaded successfully")
except Exception as e:
    print(f"Error loading EPUB: {e}")
    exit(1)

# Create a dictionary to store chapter contents
ebook = {}

# Process each item in the book
item_count = 0
chapter_num = 1

print(f"Total items in book: {len(list(book.get_items()))}")

for item in book.get_items():
    item_count += 1
    item_id = f"item_{item_count}"
    
    if item.get_type() == ebooklib.ITEM_DOCUMENT:
        try:
            content = item.get_content().decode('utf-8')
            print(f"Processing {item_id} - size: {len(content)} bytes")
            
            # Skip very small items
            if len(content) < 100:
                print(f"Skipping {item_id} - too small")
                continue
                
            # Clean the content
            text = clean_html_content(content)
            
            # Extract chapter content
            chapter_text = extract_main_chapter(text, item_id)
            
            if not chapter_text:
                print(f"No chapter content found in {item_id}")
                continue
            
            print(f"Found chapter content in {item_id} - {len(chapter_text)} bytes")
            
            # Split into sentences
            sentences = sent_tokenize(chapter_text, language='spanish')
            
            # Filter out empty sentences
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Store in dictionary
            ebook[f"chapter{chapter_num}"] = sentences
            print(f"Added chapter{chapter_num} with {len(sentences)} sentences")
            chapter_num += 1
            
        except Exception as e:
            print(f"Error processing {item_id}: {e}")

# Write to JSON file
output_path = bdir.replace('.epub', '.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(ebook, f, ensure_ascii=False, indent=2)

print(f"Extraction complete. Found {chapter_num-1} chapters.")
print(f"JSON saved to {output_path}")
print(f"Debug files saved to {debug_dir}")


