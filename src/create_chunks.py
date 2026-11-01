
import pandas as pd
from pathlib import Path
import json
import re

# Project root
project_root = Path(__file__).resolve().parent.parent

# Folders
processed_folder = project_root / "processed_data"
chunks_folder = project_root / "chunks"
chunks_folder.mkdir(exist_ok=True)

print("=" * 60)
print("PHASE 2C - CHUNK CREATION")
print("=" * 60)


# ============================================================
# PART 1 - CSV CHUNKS
# ============================================================

csv_file = processed_folder / "cleaned_procurement_data.csv"

df = pd.read_csv(
    csv_file,
    keep_default_na=False,
    na_values=["nan", "NaN", "NAN", "null", "NULL", "None"]
).fillna("")

csv_chunks = []

for index, row in df.iterrows():

    # Clean missing-value markers from individual fields
    def clean_value(value):
        value = str(value).strip()

        if value.lower() in ["", "nan", "null", "none"]:
            return ""

        return value

    award_date = clean_value(row["Award Date"])
    fiscal_year = clean_value(row["Quarter and Fiscal Year"])
    commodity = clean_value(row["Commodity Category"])
    description = clean_value(row["Contract Description"])
    organization = clean_value(row["WBG Organization"])
    selection_number = clean_value(row["Selection Number"])
    supplier = clean_value(row["Supplier"])
    country = clean_value(row["Supplier Country / Economy"])
    country_code = clean_value(row["Supplier Country / Economy Code"])
    award_amount = clean_value(row["Contract Award Amount (USD)"])
    fund_source = clean_value(row["Fund Source"])
    vpu_description = clean_value(row["VPU description"])
    award_year = clean_value(row["Contract awarded - Calendar year"])

    # Display a clear message when the award amount is missing
    if not award_amount:
        award_amount = "Amount not available"

    text = f"""
Award Date: {award_date}
Quarter and Fiscal Year: {fiscal_year}
Commodity Category: {commodity}
Contract Description: {description}
WBG Organization: {organization}
Selection Number: {selection_number}
Supplier: {supplier}
Supplier Country / Economy: {country}
Supplier Country / Economy Code: {country_code}
Contract Award Amount (USD): {award_amount}
Fund Source: {fund_source}
VPU Description: {vpu_description}
Contract Awarded - Calendar Year: {award_year}
""".strip()

    csv_chunks.append({
        "chunk_id": f"csv_{index + 1}",
        "source": "corporate_procurement_contract_awards_10-02-2026.csv",
        "source_type": "csv",
        "text": text,
        "metadata": {
            "supplier": supplier,
            "country": country,
            "organization": organization,
            "award_year": award_year
        }
    })


print(f"\nCSV records converted to chunks: {len(csv_chunks)}")


# ============================================================
# PART 2 - PDF CHUNKS
# ============================================================

pdf_text_files = [
    processed_folder / "Vendor Code of Conduct_ocr.txt",
    processed_folder / "WBG Terms and Conditions_ocr.txt"
]

pdf_chunks = []

for text_file in pdf_text_files:

    print(f"\nProcessing PDF text: {text_file.name}")

    if not text_file.exists():
        print(f"WARNING: File not found: {text_file}")
        continue

    with open(text_file, "r", encoding="utf-8") as file:
        content = file.read()

    # Split using page markers
    pages = re.split(r"--- PAGE \d+ ---", content)

    page_number = 0

    for page in pages:

        page = page.strip()

        if not page:
            continue

        page_number += 1

        pdf_chunks.append({
            "chunk_id": f"{text_file.stem}_{page_number}",
            "source": text_file.name,
            "source_type": "pdf",
            "text": page,
            "metadata": {
                "page": page_number
            }
        })

    print(f"Pages converted to chunks: {page_number}")


# ============================================================
# PART 3 - COMBINE ALL CHUNKS
# ============================================================

all_chunks = csv_chunks + pdf_chunks

output_file = chunks_folder / "all_chunks.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(all_chunks, file, indent=2, ensure_ascii=False)


# ============================================================
# PART 4 - RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CHUNK CREATION RESULTS")
print("=" * 60)

print(f"CSV chunks: {len(csv_chunks)}")
print(f"PDF chunks: {len(pdf_chunks)}")
print(f"Total chunks: {len(all_chunks)}")

print("\nSaved to:")
print(output_file)

print("\nFirst chunk:")
print(json.dumps(all_chunks[0], indent=2, ensure_ascii=False))

print("\n" + "=" * 60)
print("PHASE 2C COMPLETED")
print("=" * 60)

