import fitz
import pytesseract
from PIL import Image
from pathlib import Path
import io

# Project root
project_root = Path(__file__).resolve().parent.parent

# PDF files
pdf_files = [
    project_root / "Vendor Code of Conduct.pdf",
    project_root / "WBG Terms and Conditions.pdf"
]

# Output folder
output_folder = project_root / "processed_data"
output_folder.mkdir(exist_ok=True)

print("=" * 60)
print("PHASE 2 - PDF OCR")
print("=" * 60)

for pdf_file in pdf_files:

    print("\n" + "-" * 60)
    print(f"Processing: {pdf_file.name}")
    print("-" * 60)

    if not pdf_file.exists():
        print("ERROR: PDF file not found.")
        continue

    document = fitz.open(pdf_file)

    all_text = []

    print(f"Number of pages: {len(document)}")

    for page_number, page in enumerate(document, start=1):

        # Try normal text extraction first
        text = page.get_text().strip()

        if text:
            page_text = text
            print(f"Page {page_number}: Text extracted directly")

        else:
            # Render page as image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            image = Image.open(io.BytesIO(pix.tobytes("png")))

            # OCR
            page_text = pytesseract.image_to_string(
                image,
                lang="eng"
            ).strip()

            print(f"Page {page_number}: OCR completed")

        all_text.append(
            f"\n--- PAGE {page_number} ---\n{page_text}"
        )

    document.close()

    # Create output filename
    output_name = pdf_file.stem + "_ocr.txt"
    output_file = output_folder / output_name

    # Save OCR text
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(all_text))

    print(f"\nSaved OCR text to:")
    print(output_file)

print("\n" + "=" * 60)
print("PHASE 2 OCR COMPLETED")
print("=" * 60)