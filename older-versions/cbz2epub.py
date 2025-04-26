#!/usr/bin/env python3

import os
import zipfile
import shutil
from ebooklib import epub
from PIL import Image
import argparse

def convert_cbz_to_epub(input_file, output_file=None):
    """
    Convert a CBZ file to EPUB format
    """
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + '.epub'

    # Create a temporary directory for extraction
    temp_dir = 'temp_cbz_extract'
    os.makedirs(temp_dir, exist_ok=True)

    # Create new EPUB book
    book = epub.EpubBook()
    book.set_title(os.path.basename(input_file))
    book.set_language('en')
    book.add_author('Converted by CBZ2EPUB')

    try:
        # Extract CBZ contents
        with zipfile.ZipFile(input_file, 'r') as cbz:
            cbz.extractall(temp_dir)

        # Get all image files and sort them
        image_files = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        image_files.sort()

        # Create chapters for each image
        chapters = []
        spine = ['nav']

        for idx, img_path in enumerate(image_files, 1):
            # Create image chapter
            chapter = epub.EpubHtml(title=f'Page {idx}',
                                  file_name=f'page_{idx}.xhtml')
            
            # Convert image to JPEG if it's PNG (better compatibility)
            if img_path.lower().endswith('.png'):
                img = Image.open(img_path)
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                jpeg_path = os.path.splitext(img_path)[0] + '.jpg'
                img.save(jpeg_path, 'JPEG', quality=85)
                img_path = jpeg_path

            # Add image to EPUB
            img_name = f'images/image_{idx}.jpg'
            book.add_item(epub.EpubImage(
                uid=f'image_{idx}',
                file_name=img_name,
                media_type='image/jpeg',
                content=open(img_path, 'rb').read()
            ))

            # Create HTML for the image
            chapter.content = f'<html><body><img src="{img_name}" /></body></html>'
            book.add_item(chapter)
            chapters.append(chapter)
            spine.append(chapter)

        # Add navigation files
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        # Create table of contents
        book.toc = [(epub.Section('Pages'), chapters)]

        # Set the spine
        book.spine = spine

        # Write EPUB file
        epub.write_epub(output_file, book)

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description='Convert CBZ files to EPUB format')
    parser.add_argument('input_file', help='Input CBZ file')
    parser.add_argument('-o', '--output', help='Output EPUB file (optional)')
    args = parser.parse_args()

    convert_cbz_to_epub(args.input_file, args.output)
    print(f"Conversion complete: {args.input_file} -> {args.output or os.path.splitext(args.input_file)[0] + '.epub'}")

if __name__ == '__main__':
    main()
