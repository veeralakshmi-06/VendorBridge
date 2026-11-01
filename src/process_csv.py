
import pandas as pd
from pathlib import Path

# Project root
project_root = Path(__file__).resolve().parent.parent

# Find CSV file
csv_files = list(project_root.glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV file found.")
    raise SystemExit(1)

csv_file = csv_files[0]

print("=" * 60)
print("PHASE 2B - CSV PROCESSING")
print("=" * 60)

print(f"\nProcessing: {csv_file.name}")

# Read CSV and recognize common missing-value markers
df = pd.read_csv(
    csv_file,
    na_values=["nan", "NaN", "NAN", "null", "NULL", "None"],
    keep_default_na=True
)

print(f"Original rows: {len(df)}")
print(f"Original columns: {len(df.columns)}")

# Remove completely empty rows
df = df.dropna(how="all")

# Convert missing values to empty strings
df = df.fillna("")

# Clean text columns
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].astype(str).str.strip()

        # Clean textual missing-value markers
        df[column] = df[column].replace(
            to_replace=r"(?i)^(nan|null|none)$",
            value="",
            regex=True
        )

# Create output folder
output_folder = project_root / "processed_data"
output_folder.mkdir(exist_ok=True)

# Save cleaned CSV
output_file = output_folder / "cleaned_procurement_data.csv"

df.to_csv(output_file, index=False, encoding="utf-8")

print(f"\nCleaned rows: {len(df)}")
print(f"Saved cleaned CSV to:")
print(output_file)

print("\nFirst 5 cleaned records:")
print(df.head(5).to_string(index=False))

# Report missing values in the cleaned data
remaining_markers = df.astype(str).apply(
    lambda column: column.str.fullmatch(
        r"(?i)\s*(nan|null|none)\s*"
    )
).sum().sum()

print(f"\nRemaining textual missing-value markers: {remaining_markers}")

print("\n" + "=" * 60)
print("CSV PROCESSING COMPLETED")
print("=" * 60)

