epub_file = "epub/espagnol/DonQuijote.epub"

import ebooklib
from ebooklib import epub
import json
import re
from bs4 import BeautifulSoup
import warnings

# Optional: Suppress specific ebooklib warnings if they are not critical for this script.
# Filter by specific message if possible to avoid hiding other potentially useful warnings.
warnings.filterwarnings("ignore", message="In the future version we will turn default option ignore_ncx to True.", category=UserWarning, module="ebooklib.epub")
warnings.filterwarnings("ignore", message="This search incorrectly ignores the root element, and will be fixed in a future version.", category=FutureWarning, module="ebooklib.epub")

def text_to_sentences(text_block: str) -> list[str]:
    """
    Splits a text block into a list of sentences.
    Assumes newlines have already been handled (e.g., converted to spaces).
    """
    # Replace any remaining newlines (should be none if pre-processed correctly) with spaces
    text_block = text_block.replace('\n', ' ')
    # Normalize multiple spaces into single spaces
    text_block = re.sub(r'\s+', ' ', text_block).strip()
    
    if not text_block:
        return []

    # Regex to split sentences. It looks for common sentence terminators (. ! ?)
    # followed by a space and an uppercase letter, or end of string.
    # It tries to handle cases like "Mr. Smith" or "Dr. Who" by not splitting after a period
    # if it's followed by a space and then a lowercase letter (common in abbreviations).
    # This is a basic approach and might need refinement for complex texts.
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ])|(?<=[.!?])\s*$', text_block)
    
    # Filter out any empty strings or strings consisting only of a period
    sentences = [s.strip() for s in sentences if s and s.strip() and s.strip() != '.']
    return sentences

def extract_chapters_from_epub(epub_path):
    book = epub.read_epub(epub_path)
    chapters = {}
    all_text_parts = []

    # 1. Concatenate all document content from the EPUB
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html_content = item.get_content().decode('utf-8', errors='ignore')
        soup = BeautifulSoup(html_content, 'html.parser')
        # Get text, try to preserve paragraph structure by using newline as separator.
        # strip=False initially, will be cleaned globally later.
        text = soup.get_text(separator='\n', strip=False)
        all_text_parts.append(text)
    
    full_book_text = "\n".join(all_text_parts)

    # Normalize whitespace:
    # - Replace all newlines (and surrounding whitespace) with a single space.
    # - Collapse multiple spaces/tabs into a single space.
    # - Strip leading/trailing whitespace from the entire text.
    full_book_text = re.sub(r'\s*\n\s*', ' ', full_book_text) # Replace newlines with space
    full_book_text = re.sub(r'[ \t]+', ' ', full_book_text).strip()

    # 2. Define regex for chapter titles.
    # This regex aims to capture "Capítulo " followed by a Roman numeral or a Spanish ordinal word.
    roman_numerals_pattern = r"[IVXLCDM]+"
    # A selection of Spanish ordinal words (can be expanded if needed)
    # Includes common forms and allows for 'vigésimo primero' type constructs.
    ordinals_pattern = (
        r"primero|segundo|tercero|cuarto|quinto|sexto|s[eé]ptimo|octavo|noveno|d[eé]cimo|"
        r"und[eé]cimo|duod[eé]cimo|"
        r"decimotercero|decimocuarto|decimoquinto|decimosexto|decimos[eé]ptimo|decimoctavo|decimonoveno|" # Common compound forms
        r"vig[eé]simo(?:\s+(?:primero|segundo|tercero|cuarto|quinto|sexto|s[eé]ptimo|octavo|noveno|d[eé]cimo))?|"
        r"trig[eé]simo(?:\s+(?:primero|segundo|tercero|cuarto|quinto))?|cuadrag[eé]simo|quincuag[eé]simo" # Extend as needed
    )
    
    # Combined pattern for the part after "Capítulo "
    # This capturing group will extract the numeral/ordinal part.
    chapter_number_part_pattern = f"({roman_numerals_pattern}|{ordinals_pattern})"
    
    # Full pattern to split by: (Capítulo + space + NUMBER_PART)
    # The outer group captures the full "Capítulo X" heading.
    full_chapter_heading_pattern = rf"(Capítulo\s+{chapter_number_part_pattern})"

    # 3. Split the text by chapter titles using the regex.
    # re.split with capturing groups includes the delimiters in the result.
    # Result: [text_before_first_cap, full_cap1_match, cap1_num_match, content1, full_cap2_match, cap2_num_match, content2, ...]
    parts = re.split(full_chapter_heading_pattern, full_book_text, flags=re.IGNORECASE)

    # Handle text before the first "Capítulo" (e.g., preface, introduction)
    if parts and parts[0].strip():
        pre_chapter_text = parts[0].strip()
        if len(pre_chapter_text) > 50: # Only add if it's substantial
            chapters["Introducción / Prólogo"] = text_to_sentences(pre_chapter_text)

    # Iterate through the parts to extract chapter titles and their content
    # Start idx at 1 to skip the initial pre-chapter text.
    # Each chapter segment is: parts[idx] (full title), parts[idx+1] (number part), parts[idx+2] (content)
    idx = 1
    while idx + 2 < len(parts):
        chapter_title = parts[idx].strip()
        # chapter_number_extracted = parts[idx+1].strip() # This is the isolated number/ordinal, can be used if needed
        chapter_content = parts[idx+2].strip()

        if chapter_title: # Ensure there's a title
            # Handle potential duplicate chapter titles (e.g. if regex matches oddly or structure is unusual)
            original_title = chapter_title
            counter = 1
            while chapter_title in chapters:
                chapter_title = f"{original_title} ({counter})"
                counter += 1
            chapters[chapter_title] = text_to_sentences(chapter_content)
        
        idx += 3 # Move to the next chapter segment (title, number_part, content)
            
    return chapters

if __name__ == "__main__":
    epub_path = "epub/espagnol/DonQuijote.epub"
    chapters_content = extract_chapters_from_epub(epub_path)

    output_json_path = "don_quijote_chapters.json"
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(chapters_content, json_file, ensure_ascii=False, indent=4)

    print(f"Extracted {len(chapters_content)} chapters and saved to {output_json_path}")

    # Optional: Print a snippet of the first few chapters to verify
    # count = 0
    # for title, content_sentences in chapters_content.items():
    #     print(f"Chapter: {title}")
    #     if content_sentences: # Check if there are any sentences
    #         print(f"First sentence: {content_sentences[0]}")
    #         print(f"Number of sentences: {len(content_sentences)}\n")
    #     else:
    #         print("This chapter has no content or failed to parse sentences.\n")
    #     count += 1
    #     if count >= 3: # Print only the first 3 chapters as a sample
    #         break

