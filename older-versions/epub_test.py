import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re

book = epub.read_epub('epub/espagnol/Plateroyyo.epub')

# Function to debug the structure of the ePub file with content samples
def debug_epub_structure(book):
    print("\n===== EPUB STRUCTURE =====")
    
    # Get all items
    all_items = list(book.get_items())
    print(f"Total items: {len(all_items)}")
    
    # Get document items
    doc_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    print(f"Document items: {len(doc_items)}")
    
    # Print spine items
    spine_items = book.spine
    print(f"Spine items: {len(spine_items)}")
    
    # Print details of each document item with content sample
    print("\nDocument items details:")
    for idx, item in enumerate(doc_items):
        content = item.get_content().decode('utf-8', errors='replace')
        soup = BeautifulSoup(content, 'html.parser')
        title = soup.title.string if soup.title else 'No title'
        text_content = soup.get_text().strip()
        content_sample = text_content[:100] + "..." if len(text_content) > 100 else text_content
        print(f"Item #{idx}: ID={item.get_id()}, Name={item.get_name()}, Title={title}")
        print(f"   Content sample: {content_sample}")
        
    # Print details of spine items with content
    print("\nSpine items details:")
    for idx, (idref, linear) in enumerate(spine_items):
        item = book.get_item_with_id(idref)
        content = item.get_content().decode('utf-8', errors='replace')
        soup = BeautifulSoup(content, 'html.parser')
        text_content = soup.get_text().strip()
        content_sample = text_content[:100] + "..." if len(text_content) > 100 else text_content
        print(f"Spine #{idx}: idref={idref}, linear={linear}")
        print(f"   Content sample: {content_sample}")

# Function to find content at a specific CFI
def get_content_at_cfi(book, cfi):
    print(f"cfi: {cfi}")
    try:
        # Parse the CFI properly
        # Format example: epubcfi(/6/4!/4/90/2/6/2/1:0)
        
        # Extract the spine component from the path part
        match = re.search(r'/(\d+)/(\d+)!', cfi)
        if not match:
            raise ValueError("Could not find proper spine path in CFI")
            
        # In the CFI format, the first number after the root element (/6)
        # typically refers to the body element, and the second number
        # refers to the spine item index
        spine_index = int(match.group(2)) - 1  # CFI is 1-based, our list is 0-based
        print(f"spine_index: {spine_index}")
        
        # Try using the spine directly instead of document items
        spine_items = book.spine
        if spine_index < 0 or spine_index >= len(spine_items):
            print(f"Warning: Spine index {spine_index} out of range for spine (0-{len(spine_items)-1}), trying document items")
            
            # Get document items as fallback
            items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
            
            if spine_index < 0 or spine_index >= len(items):
                raise ValueError(f"Spine index {spine_index} out of range for document items (0-{len(items)-1})")
                
            item = items[spine_index]
        else:
            # Get the item from the spine
            idref = spine_items[spine_index][0]
            item = book.get_item_with_id(idref)
            
        content = item.get_content().decode('utf-8', errors='replace')
        
        # Clean up the HTML content for better readability
        soup = BeautifulSoup(content, 'html.parser')
        clean_content = soup.get_text()
        
        return {
            'content': clean_content,
            'item_id': item.get_id(),
            'item_name': item.get_name(),
            'spine_position': spine_index + 1,
            'html': str(soup)  # Add the HTML for inspection
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to get content by spine index directly
def get_content_by_spine_index(book, index):
    try:
        spine_items = book.spine
        if index < 0 or index >= len(spine_items):
            raise ValueError(f"Index {index} out of range (0-{len(spine_items)-1})")
        
        idref = spine_items[index][0]
        item = book.get_item_with_id(idref)
        content = item.get_content().decode('utf-8', errors='replace')
        
        soup = BeautifulSoup(content, 'html.parser')
        return {
            'content': soup.get_text(),
            'item_id': item.get_id(),
            'item_name': item.get_name(),
            'spine_position': index + 1,
            'html': str(soup)
        }
    except Exception as e:
        print(f"Error accessing spine index {index}: {e}")
        return None

# Function to find a specific text passage in the ePub with exact matching
def find_text_in_epub(book, search_text, case_sensitive=False, max_results=10):
    print(f"\n===== SEARCHING FOR: {search_text[:50]}... =====")
    print(f"Case sensitive: {case_sensitive}")
    
    # Get all document items
    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    
    # Also check spine items specifically
    spine_items = book.spine
    
    found_results = []
    search_text_clean = search_text.strip()
    
    # Search in all document items
    print("Searching in document items...")
    for idx, item in enumerate(items):
        content = item.get_content().decode('utf-8', errors='replace')
        content_for_search = content if case_sensitive else content.lower()
        search_for = search_text_clean if case_sensitive else search_text_clean.lower()
        
        if search_for in content_for_search:
            print(f"Found in document item #{idx}: {item.get_id()}")
            
            # Parse the HTML to get context
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find paragraphs containing the text
            for p_idx, p in enumerate(soup.find_all(['p', 'div'])):
                p_text = p.get_text()
                p_text_for_search = p_text if case_sensitive else p_text.lower()
                
                if search_for in p_text_for_search:
                    print(f"Found in paragraph/div #{p_idx}")
                    
                    found_results.append({
                        'item_type': 'document',
                        'item_index': idx,
                        'item_id': item.get_id(),
                        'item_name': item.get_name(),
                        'paragraph_index': p_idx,
                        'content': p_text.strip(),
                        'html': str(p)
                    })
                    
                    # If we've found the maximum number of results, stop searching
                    if len(found_results) >= max_results:
                        return found_results

    # If we haven't found enough results in document items, search in spine items
    if len(found_results) < max_results:
        print("Searching in spine items...")
        for idx, (idref, linear) in enumerate(spine_items):
            item = book.get_item_with_id(idref)
            content = item.get_content().decode('utf-8', errors='replace')
            content_for_search = content if case_sensitive else content.lower()
            
            if search_for in content_for_search:
                print(f"Found in spine item #{idx}: {idref}")
                
                # Parse the HTML to get context
                soup = BeautifulSoup(content, 'html.parser')
                
                # Find paragraphs containing the text
                for p_idx, p in enumerate(soup.find_all(['p', 'div'])):
                    p_text = p.get_text()
                    p_text_for_search = p_text if case_sensitive else p_text.lower()
                    
                    if search_for in p_text_for_search:
                        print(f"Found in paragraph/div #{p_idx}")
                        
                        found_results.append({
                            'item_type': 'spine',
                            'item_index': idx,
                            'item_id': idref,
                            'item_name': item.get_name(),
                            'paragraph_index': p_idx,
                            'content': p_text.strip(),
                            'html': str(p)
                        })
                        
                        # If we've found the maximum number of results, stop searching
                        if len(found_results) >= max_results:
                            return found_results
    
    return found_results

# Search for the exact passage
search_text = "EN mi duermevela matinal, me malhumora una endiablada chillería de chiquillos"
results = find_text_in_epub(book, search_text, case_sensitive=False, max_results=5)

# Display search results
if results:
    print("\n===== SEARCH RESULTS =====")
    for idx, result in enumerate(results):
        print(f"\nResult #{idx+1}:")
        print(f"Found in {result['item_type']} item #{result['item_index']}: {result['item_id']}")
        print(f"Paragraph #{result['paragraph_index']}")
        print(f"Full content: {result['content']}")
        
        # Generate a potential CFI
        if result['item_type'] == 'spine':
            spine_idx = result['item_index'] + 1
            para_idx = result['paragraph_index'] + 1
            approx_cfi = f"epubcfi(/6/{spine_idx}!/4/2/{para_idx}:0)"
            print(f"Approximate CFI: {approx_cfi}")
else:
    print("\nNo matching text found in the ePub. Trying alternative search...")
    
    # Try with a distinct paragraph that should be unique
    search_text2 = "EN mi duermevela matinal, me malhumora una endiablada chillería de chiquillos. Por fin, sin poder dormir más"
    results2 = find_text_in_epub(book, search_text2, case_sensitive=False, max_results=5)
    
    if results2:
        print("\n===== ALTERNATIVE SEARCH RESULTS =====")
        for idx, result in enumerate(results2):
            print(f"\nResult #{idx+1}:")
            print(f"Found in {result['item_type']} item #{result['item_index']}: {result['item_id']}")
            print(f"Paragraph #{result['paragraph_index']}")
            print(f"Full content: {result['content']}")
            
            # Generate a potential CFI
            if result['item_type'] == 'spine':
                spine_idx = result['item_index'] + 1
                para_idx = result['paragraph_index'] + 1
                approx_cfi = f"epubcfi(/6/{spine_idx}!/4/2/{para_idx}:0)"
                print(f"Approximate CFI: {approx_cfi}")
    else:
        print("\nStill no matches found. Let's try to find any paragraph containing 'duermevela'...")
        
        search_text3 = "duermevela"
        results3 = find_text_in_epub(book, search_text3, case_sensitive=False, max_results=5)
        
        if results3:
            print("\n===== KEYWORD SEARCH RESULTS =====")
            for idx, result in enumerate(results3):
                print(f"\nResult #{idx+1}:")
                print(f"Found in {result['item_type']} item #{result['item_index']}: {result['item_id']}")
                print(f"Paragraph #{result['paragraph_index']}")
                print(f"Full content: {result['content']}")
                
                # Generate a potential CFI
                if result['item_type'] == 'spine':
                    spine_idx = result['item_index'] + 1
                    para_idx = result['paragraph_index'] + 1
                    approx_cfi = f"epubcfi(/6/{spine_idx}!/4/2/{para_idx}:0)"
                    print(f"Approximate CFI: {approx_cfi}")

# Debug the ePub structure
debug_epub_structure(book)
