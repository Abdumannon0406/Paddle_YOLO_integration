from pdf2image import convert_from_path
import os
from pathlib import Path

pdf_folder = Path("Datasets")
pdf_paths = pdf_folder.glob('*.pdf')
output_dir = Path("Images")

output_dir.mkdir(exist_ok=True)

for path in pdf_paths:
    pdf_name = path.stem  # filename without extension

    # Convert PDF pages to images
    pages = convert_from_path(path)

    # Save each page as an image
    for i, page in enumerate(pages):
        image_filename = f"{pdf_name}_page_{i+1}.jpg"
        image_path = output_dir / image_filename
        page.save(image_path, "JPEG")
        print(f"Saved: {image_path}")
