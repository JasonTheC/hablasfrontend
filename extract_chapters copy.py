import os
import json
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import nltk
from bs4 import XMLParsedAsHTMLWarning
import warnings
import re # For cleaning up titles for keys

# Download sentence tokenizer models if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def get_chapter_title(item_content):
    """
    Attempts to extract a chapter title from the item content.
    Looks for h1, h2, or h3 tags.
    """
    soup = BeautifulSoup(item_content, 'html.parser')
    for tag_name in ['h1', 'h2', 'h3']:
        tag = soup.find(tag_name)
        if tag and tag.get_text(strip=True):
            return tag.get_text(strip=True)
    return None

def flatten_toc(toc_items, current_level=0):
    """ 
    Recursively flattens the ToC structure from ebooklib, 
    handling Sections and Links.
    Returns a list of tuples: (title, href, level)
    """
    flat_list = []
    if not isinstance(toc_items, list):
        # If a single Link or Section object is passed (or something else not a list)
        # wrap it in a list to proceed, or handle if it's not a valid ToC item type.
        if isinstance(toc_items, (ebooklib.epub.Link, ebooklib.epub.Section, tuple)):
            toc_items = [toc_items]
        else:
            # Not a processable ToC item or list of items
            # print(f"DEBUG: flatten_toc received non-list, non-ToC item: {type(toc_items)}")
            return flat_list
            
    for item in toc_items:
        if isinstance(item, ebooklib.epub.Link):
            if hasattr(item, 'title') and hasattr(item, 'href') and item.href:
                # Ensure title is a string, not None
                title = item.title if item.title is not None else ""
                flat_list.append((str(title), item.href, current_level))
        elif isinstance(item, ebooklib.epub.Section):
            # A section itself might be a navigable item OR a container.
            section_title_str = str(item.title) if hasattr(item, 'title') and item.title is not None else ""
            if hasattr(item, 'href') and item.href:
                flat_list.append((section_title_str, item.href, current_level))
            
            if hasattr(item, 'children') and item.children:
                flat_list.extend(flatten_toc(item.children, current_level + 1))
        
        elif isinstance(item, tuple):
            if not item: continue

            if len(item) > 1 and isinstance(item[1], list):
                item_zero_title_str = ""
                if isinstance(item[0], (ebooklib.epub.Section, ebooklib.epub.Link)) and \
                   hasattr(item[0], 'title') and item[0].title is not None:
                    item_zero_title_str = str(item[0].title)
                elif isinstance(item[0], str):
                    item_zero_title_str = item[0]
                
                if (isinstance(item[0], (ebooklib.epub.Section, ebooklib.epub.Link))) and \
                   hasattr(item[0], 'href') and item[0].href:
                    flat_list.append((item_zero_title_str, item[0].href, current_level))
                
                flat_list.extend(flatten_toc(item[1], current_level + 1))
            
            elif len(item) >= 2 and isinstance(item[0], str) and isinstance(item[1], str):
                flat_list.append((item[0], item[1], current_level))
            
            elif len(item) == 1 and isinstance(item[0], (ebooklib.epub.Link, ebooklib.epub.Section)):
                flat_list.extend(flatten_toc([item[0]], current_level))
    return flat_list

def extract_text_from_element(start_element, end_element_id=None):
    """
    Extracts all text from <p> tags starting from start_element 
    until an element with end_element_id is encountered or start_element's parent ends.
    """
    text_parts = []
    current_element = start_element
    while current_element:
        if end_element_id and current_element.name != 'script' and current_element.name != 'style': # Avoid recursing into these
            # Check if current_element itself or any of its children is the end_element
            if current_element.get('id') == end_element_id:
                break
            # Also check descendants for the end_element_id, but be careful not to stop prematurely
            # if the end_element_id is nested deep within a paragraph we want to include.
            # For now, a simpler check: if the end_id is a sibling or an id of current element, stop.
            # More robust checking would involve looking ahead in siblings only.

        if current_element.name == 'p':
            text_parts.append(current_element.get_text(strip=True))
        
        # Move to the next sibling
        # If end_element_id is found in a sibling, we stop before processing it.
        next_sibling = current_element.find_next_sibling()
        if end_element_id and next_sibling and next_sibling.get('id') == end_element_id:
            break
        current_element = next_sibling
        
    return ' '.join(filter(None, text_parts))

def get_content_item_by_href(book, href_target):
    """
    Retrieves a book item (like ITEM_DOCUMENT) by its href.
    Handles cases where href might have an anchor and leading ../
    """
    # Remove anchor part for item lookup
    clean_href = href_target.split('#')[0]
    
    # Normalize href by removing any leading relative path components like ../
    # This is important because item.file_name in ebooklib is usually relative to OPF root
    normalized_item_href = clean_href
    while normalized_item_href.startswith('../'):
        normalized_item_href = normalized_item_href[3:]

    # Try to get item by this normalized href
    item = book.get_item_with_href(normalized_item_href) # ebooklib handles this well
    if item:
        return item
    
    # Fallback: Sometimes hrefs in ToC might be relative to the ToC file itself,
    # or simply not directly match item.file_name. Iterate to find.
    # This part might be redundant if get_item_with_href is robust enough with normalized_item_href
    for book_item in book.get_items():
        if hasattr(book_item, 'file_name') and book_item.file_name.endswith(clean_href):
            return book_item
            
    return None # Should ideally use book_items_by_href created in the main function for direct lookup

def sanitize_title_for_key(title):
    """Sanitizes a title to be a valid and clean JSON key."""
    if not title: title = "untitled_section"
    # Remove or replace special characters, convert to lowercase, replace spaces
    key = title.lower()
    key = re.sub(r'[^a-z0-9_\s-]', '', key) # Remove non-alphanumeric (except underscore, space, hyphen)
    key = re.sub(r'[\s-]+', '_', key) # Replace spaces/hyphens with underscore
    key = key.strip('_')
    return key if key else "untitled_chapter"

def extract_chapters_heuristically(book):
    """
    Fallback: Extracts chapters by treating each content document in the spine as a chapter.
    Returns a list of dicts: [{'title': str, 'sentences': list, 'is_explicit_preface': bool, 'order': int}]
    """
    print("  Attempting heuristic chapter extraction (each content file as a chapter).")
    heuristic_sections = []
    item_order = 0

    # Use book.spine to get items in reading order if possible
    # book.spine contains IDs, we need to get items from these IDs
    spine_items = []
    if hasattr(book, 'spine') and book.spine:
        for item_id_tuple in book.spine:
            item_id = item_id_tuple[0] # spine is list of tuples like [('id_of_item', 'linear_yes_or_no')]
            item = book.get_item_with_id(item_id)
            if item and item.get_type() == ebooklib.ITEM_DOCUMENT:
                spine_items.append(item)
    
    # If spine is empty or didn't yield documents, fall back to all documents in any order
    # (though ebooklib usually gives them in manifest order)
    items_to_process = spine_items if spine_items else list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    for item in items_to_process:
        content = item.get_content()
        soup = BeautifulSoup(content, 'html.parser')
        
        title_text = get_chapter_title(content) # Existing function to find h1/h2 etc.
        if not title_text: # Fallback title if no h-tag found
            title_text = item.file_name # Use filename as a fallback title

        text_parts = []
        for p_tag in soup.find_all('p'):
            paragraph_text = p_tag.get_text(strip=True)
            if paragraph_text:
                text_parts.append(paragraph_text)
        
        if not text_parts:
            continue

        full_text = ' '.join(text_parts)
        sentences = nltk.sent_tokenize(full_text)
        
        if not sentences:
            continue

        item_order += 1
        is_explicit_preface = False # Heuristic method doesn't strongly assert explicit prefaces from titles alone here
                                  # The main logic will make the first found section the preface.
        if title_text: # Check if the heuristically found title suggests preface
            preface_keywords = ["prefácio", "preface", "introdução", "intro", "prólogo", "prologue", "ao leitor"]
            if any(keyword in str(title_text).lower() for keyword in preface_keywords):
                is_explicit_preface = True
        
        heuristic_sections.append({
            'title': str(title_text),
            'sentences': sentences, 
            'is_explicit_preface': is_explicit_preface, 
            'order': item_order
        })
    return heuristic_sections

def extract_sentences_from_epub(epub_path):
    """
    Extracts chapters and their sentences from an EPUB file using ToC or heuristics.
    """
    book = epub.read_epub(epub_path)
    raw_toc_entries = [] # Stores (title, href, level) from ToC parsing

    if book.toc:
        raw_toc_entries = flatten_toc(book.toc)
    
    if not raw_toc_entries:
        nav_items = list(book.get_items_of_type(ebooklib.ITEM_NAVIGATION))
        if nav_items:
            nav_item = nav_items[0]
            nav_soup = BeautifulSoup(nav_item.get_content(), 'html.parser')
            epub_toc_nav = nav_soup.find('nav', attrs={'epub:type': 'toc'})
            if epub_toc_nav:
                for a_tag in epub_toc_nav.find_all('a', href=True):
                    raw_toc_entries.append((a_tag.get_text(strip=True), a_tag['href'], 0))
            else:
                for list_tag in nav_soup.find_all(['ol', 'ul']):
                    for a_tag in list_tag.find_all('a', href=True):
                        raw_toc_entries.append((a_tag.get_text(strip=True), a_tag['href'], 0))

    filtered_toc_entries = []
    if raw_toc_entries:
        print(f"  ToC found for {os.path.basename(epub_path)}. Processing ToC entries.")
        seen_hrefs = set()
        for title, href, level in raw_toc_entries:
            cleaned_title = str(title).strip()
            if not cleaned_title or len(cleaned_title) < 1: continue # Allow even single char titles if they link somewhere
            if "license" in cleaned_title.lower() or "project gutenberg" in cleaned_title.lower(): continue
            
            normalized_href = href
            while normalized_href.startswith('../'):
                normalized_href = normalized_href[3:]
            
            if normalized_href not in seen_hrefs:
                 # Skip structural titles from Memorias Posthumas based on previous observation
                if epub_path.endswith("MemoriasPosthumasdeBrazCubas.epub") and \
                   (cleaned_title.upper() == cleaned_title and len(cleaned_title.split()) < 4 and level < 2) and \
                   cleaned_title not in ["AO LEITOR", "FIM"] : # Allow AO LEITOR and FIM even if all caps
                    # print(f"DEBUG: Skipping likely structural title: '{cleaned_title}' for Memorias Posthumas")
                    continue
                filtered_toc_entries.append((cleaned_title, normalized_href, level))
                seen_hrefs.add(normalized_href)
    
    # If ToC processing yielded usable entries, use them. Otherwise, try heuristics.
    sections_to_process = [] # This will be list of {'title', 'sentences', 'is_explicit_preface', 'order'}

    if filtered_toc_entries:
        # print(f"--- Using Filtered & Unique ToC for {os.path.basename(epub_path)} ---")
        # for i, (title, href, level) in enumerate(filtered_toc_entries):
        #     print(f"  {i}. Level {level} - Title: '{title}', Href: '{href}'")

        # Prepare book items map for quick lookup by their clean href (relative to OPF root)
        # item.file_name is the key ebooklib uses internally that matches manifest hrefs.
        book_items_by_href_map = {item.file_name: item for item in book.get_items()}
        # Also add items by their ID for fallback if hrefs are weird
        # book_items_by_id_map = {item.id: item for item in book.get_items()}

        order_counter = 0
        for i, (title, href, level) in enumerate(filtered_toc_entries):
            file_path_part = href.split('#')[0]
            anchor_part = href.split('#')[1] if '#' in href else None
            
            # Try to get content item using normalized file_path_part
            content_item = book_items_by_href_map.get(file_path_part)
            # Fallback if href had relative paths that were stripped, try original if needed or search more broadly
            if not content_item:
                # Try original href part if different
                original_file_path_part = href.split('#')[0] # href before normalization for seen_hrefs
                if original_file_path_part != file_path_part:
                    content_item = book_items_by_href_map.get(original_file_path_part)
            if not content_item: # Try get_item_with_href as a broader search
                 content_item = book.get_item_with_href(file_path_part)

            if not content_item or content_item.get_type() != ebooklib.ITEM_DOCUMENT:
                # print(f"Warning: Could not find document item for href '{file_path_part}' (Title: '{title}'). Skipping.")
                continue

            soup = BeautifulSoup(content_item.get_content(), 'html.parser')
            start_node = None
            if anchor_part:
                start_node = soup.find(id=anchor_part)
            
            if not start_node: 
                start_node = soup.body or soup.find(['h1','h2','h3','p','div'])
            
            if not start_node:
                continue

            next_chapter_anchor_in_same_file = None
            if i + 1 < len(filtered_toc_entries):
                _next_title, next_href, _next_level = filtered_toc_entries[i+1]
                next_file_path_part = next_href.split('#')[0]
                if next_file_path_part == file_path_part and '#' in next_href:
                    next_chapter_anchor_in_same_file = next_href.split('#')[1]
            
            chapter_text = extract_text_from_element(start_node, next_chapter_anchor_in_same_file)
            
            if chapter_text:
                sentences = nltk.sent_tokenize(chapter_text)
                if sentences:
                    order_counter += 1
                    is_explicit_preface = False
                    preface_keywords = ["prefácio", "preface", "introdução", "intro", "prólogo", "prologue", "ao leitor"]
                    if any(keyword in title.lower() for keyword in preface_keywords):
                        is_explicit_preface = True
                    sections_to_process.append({
                        'title': title, 
                        'sentences': sentences, 
                        'is_explicit_preface': is_explicit_preface, 
                        'order': order_counter
                    })
    else: # No usable ToC entries, try heuristic approach
        print(f"  No usable ToC entries found for {os.path.basename(epub_path)}. Attempting heuristic extraction.")
        sections_to_process = extract_chapters_heuristically(book)
        if not sections_to_process:
            print(f"  Heuristic extraction also failed to find content for {os.path.basename(epub_path)}.")
            return {}

    # Final JSON structuring (preface, chapter1, chapter2, ...)
    chapters_content = {}
    preface_key_used = False
    chapter_counter = 1
    
    # Sort sections by order, just in case (heuristic might not preserve strict ToC order if mixed sources)
    sections_to_process.sort(key=lambda x: x['order'])

    for section_data in sections_to_process:
        title = section_data['title']
        sentences = section_data['sentences']
        is_explicit_preface_flag = section_data['is_explicit_preface']

        final_key = ""
        if is_explicit_preface_flag and not preface_key_used:
            final_key = "preface"
            preface_key_used = True
        elif not preface_key_used and not chapters_content: # First item becomes preface if none explicit
            final_key = "preface"
            preface_key_used = True
        else:
            final_key = f"chapter{chapter_counter}"
            chapter_counter += 1
        
        # Ensure key uniqueness if somehow conflicts (e.g. multiple explicit prefaces)
        original_final_key = final_key
        suffix = 1
        while final_key in chapters_content:
            final_key = f"{original_final_key}_{suffix}"
            suffix += 1
        chapters_content[final_key] = sentences

    return chapters_content

def main():
    epub_dir = "epub/portuguese/" # Corrected path
    book_files = ["MemoriasPosthumasdeBrazCubas.epub", "LAlchimista.epub"]
    all_books_data = {}

    for book_file in book_files:
        epub_path = os.path.join(epub_dir, book_file)
        print(f"Processing {epub_path}...")
        if not os.path.exists(epub_path):
            print(f"Error: File not found at {epub_path}")
            continue
        
        # Extract book name for JSON key
        book_name = os.path.splitext(book_file)[0] 
        
        book_specific_data = {} # For individual book JSON
        try:
            # The extract_sentences_from_epub now returns the desired structure directly
            book_specific_data = extract_sentences_from_epub(epub_path)
        except Exception as e:
            print(f"Error processing {book_file}: {e}")
            book_specific_data = {"error": str(e)}

        # Save each book's data to its own JSON file
        # The output structure per book is {preface: [...], chapter1: [...]}
        output_json_path = os.path.join(epub_dir, f"{book_name}.json") # Place JSON in the same dir
        # output_json_path = f"{book_name}.json" # If you want JSON in the root script directory

        print(f"Saving data for {book_name} to {output_json_path}...")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            # Dump the book_specific_data directly, not all_books_data[book_name]
            json.dump(book_specific_data, f, ensure_ascii=False, indent=4)
    
    # The message below is a bit misleading if we save per book,
    # maybe print a summary or just remove if it's clear from per-book messages.
    # print(f"Successfully extracted chapters to {output_json_path}") # output_json_path would be the last book's path
    print("Processing complete.")

if __name__ == "__main__":
    main() 