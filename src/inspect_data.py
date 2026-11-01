import os
import pandas as pd
import fitz


PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


print("=" * 60)
print("AI-POWERED SUPPLY CHAIN & VENDOR DOCUMENTATION ASSISTANT")
print("PHASE 1 - DATASET INSPECTION")
print("=" * 60)


# ============================================================
# CSV INSPECTION
# ============================================================

print("\n\n" + "=" * 60)
print("CSV DATASET INSPECTION")
print("=" * 60)

csv_files = [
    file for file in os.listdir(PROJECT_FOLDER)
    if file.lower().endswith(".csv")
]

if not csv_files:
    print("No CSV file found.")
else:

    csv_path = os.path.join(PROJECT_FOLDER, csv_files[0])

    print("\nCSV File:")
    print(csv_files[0])

    try:
        df = pd.read_csv(csv_path)

        print("\nNumber of Rows:", df.shape[0])
        print("Number of Columns:", df.shape[1])

        print("\nColumn Names:")
        for column in df.columns:
            print("-", column)

        print("\nFirst 5 Rows:")
        print(df.head().to_string())

    except Exception as e:
        print("\nError while reading CSV:")
        print(e)


# ============================================================
# PDF INSPECTION
# ============================================================

print("\n\n" + "=" * 60)
print("PDF DATASET INSPECTION")
print("=" * 60)

pdf_files = [
    file for file in os.listdir(PROJECT_FOLDER)
    if file.lower().endswith(".pdf")
]

if not pdf_files:
    print("No PDF files found.")
else:

    for pdf_file in pdf_files:

        pdf_path = os.path.join(PROJECT_FOLDER, pdf_file)

        print("\n" + "-" * 60)
        print("PDF File:", pdf_file)
        print("-" * 60)

        try:

            document = fitz.open(pdf_path)

            print("Number of Pages:", len(document))

            full_text = ""

            for page_number, page in enumerate(document):

                text = page.get_text()

                full_text += text + "\n"

            if full_text.strip():

                print("\nFirst 2000 Characters:")
                print(full_text[:2000])

            else:

                print("\nWARNING: No extractable text found.")

            document.close()

        except Exception as e:

            print("\nError while reading PDF:")
            print(e)


print("\n\n" + "=" * 60)
print("PHASE 1 INSPECTION COMPLETED")
print("=" * 60)