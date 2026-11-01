
import json
import os
import re
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# PHASE 5: RAG PIPELINE
# AI-Powered Supply Chain & Vendor Documentation Assistant
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_STORE = PROJECT_ROOT / "vector_store"
INDEX_FILE = VECTOR_STORE / "faiss_index.bin"
METADATA_FILE = VECTOR_STORE / "chunks_metadata.json"

PROCESSED_DATA = PROJECT_ROOT / "processed_data"
CSV_FILE = PROCESSED_DATA / "cleaned_procurement_data.csv"

ENV_FILE = PROJECT_ROOT / ".env"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"
TOP_K = 5


# ============================================================
# PART 1: LOAD CONFIGURATION
# ============================================================

load_dotenv(ENV_FILE)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found.\n"
        "Check that your project .env file contains:\n"
        "GEMINI_API_KEY=your_api_key"
    )

client = genai.Client(api_key=API_KEY)


# ============================================================
# PART 2: LOAD DATA AND MODELS
# ============================================================

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_resources():
    print("Loading FAISS index...")

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}\n"
            "Run your vector-store building script first."
        )

    loaded_index = faiss.read_index(str(INDEX_FILE))

    print("Loading chunk metadata...")

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Chunk metadata not found: {METADATA_FILE}"
        )

    loaded_metadata = load_json(METADATA_FILE)

    # Support metadata files stored as a list or a wrapper object.
    if isinstance(loaded_metadata, dict):
        for key in ["chunks", "metadata", "documents", "records"]:
            if isinstance(loaded_metadata.get(key), list):
                loaded_metadata = loaded_metadata[key]
                break

    if not isinstance(loaded_metadata, list):
        raise ValueError(
            "Unexpected metadata structure. "
            "Expected a list of chunks or a dictionary containing "
            "a chunks/metadata/documents/records list."
        )

    print("Loading procurement CSV...")

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned procurement CSV not found: {CSV_FILE}"
        )

    loaded_df = pd.read_csv(CSV_FILE)

    print("Loading embedding model...")

    loaded_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print(f"CSV records loaded: {len(loaded_df):,}")
    print(f"FAISS vectors loaded: {loaded_index.ntotal:,}")
    print(f"Metadata chunks loaded: {len(loaded_metadata):,}")

    if loaded_index.ntotal != len(loaded_metadata):
        raise ValueError(
            "The FAISS index and metadata have different sizes: "
            f"{loaded_index.ntotal} vectors versus "
            f"{len(loaded_metadata)} metadata entries. "
            "Rebuild the index and metadata from the same chunks."
        )

    print("All resources loaded successfully.\n")

    return loaded_index, loaded_metadata, loaded_df, loaded_model


index, metadata, df, embedding_model = load_resources()


# ============================================================
# PART 3: COLUMN DETECTION
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    if not isinstance(value, str):
        if pd.isna(value):
            return ""

    return " ".join(str(value).strip().lower().split())


def find_column(possible_names):
    """Find a CSV column using exact or partial name matching."""

    normalized_columns = {
        normalize_text(column): column
        for column in df.columns
    }

    for name in possible_names:
        normalized_name = normalize_text(name)

        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    for name in possible_names:
        normalized_name = normalize_text(name)

        for normalized_column, original_column in (
            normalized_columns.items()
        ):
            if normalized_name in normalized_column:
                return original_column

    return None


SUPPLIER_COLUMN = find_column([
    "supplier",
    "supplier name",
])

AWARD_DATE_COLUMN = find_column([
    "Award Date",
    "award date",
])

YEAR_COLUMN = find_column([
    "Contract Awarded - Calendar Year",
    "calendar year",
    "award year",
    "year",
])

AMOUNT_COLUMN = find_column([
    "Contract Award Amount (USD)",
    "amount USD",
    "contract award amount",
    "award amount",
])

if SUPPLIER_COLUMN is None:
    raise ValueError(
        f"Supplier column not found. CSV columns: {list(df.columns)}"
    )

if AMOUNT_COLUMN is None:
    raise ValueError(
        f"Amount column not found. CSV columns: {list(df.columns)}"
    )


# ============================================================
# PART 4: CSV HELPER FUNCTIONS
# ============================================================

def make_direct_source(description):
    return [{
        "source": "cleaned_procurement_data.csv",
        "type": "csv",
        "description": description,
    }]


def get_supplier_names():
    values = (
        df[SUPPLIER_COLUMN]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return sorted(
        {
            name
            for name in values
            if name and name.lower() != "nan"
        },
        key=str.lower,
    )


def find_named_supplier(question):
    """Find a complete supplier name present in the CSV."""

    normalized_question = normalize_text(question)

    for supplier in sorted(
        get_supplier_names(),
        key=len,
        reverse=True,
    ):
        normalized_supplier = normalize_text(supplier)

        if not normalized_supplier:
            continue

        pattern = (
            r"(?<!\w)"
            + re.escape(normalized_supplier)
            + r"(?!\w)"
        )

        if re.search(pattern, normalized_question):
            return supplier

    return None


def get_supplier_records(supplier_name):
    supplier_values = (
        df[SUPPLIER_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    mask = (
        supplier_values.map(normalize_text)
        == normalize_text(supplier_name)
    )

    return df[mask].copy()


def format_amount(value):
    amount = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(amount):
        return "Amount unavailable"

    return f"${amount:,.2f}"


def is_count_question(question):
    """Identify questions asking for the dataset's record count."""

    q = normalize_text(question)

    # Handle the most common count questions directly.
    count_patterns = [
        r"\bhow many\b.*\b(records?|rows?|entries|contracts?|awards?)\b",
        r"\bhow many\b.*\b(dataset|procurement)\b",
        r"\b(number|count)\b.*\b(records?|rows?|entries|contracts?|awards?)\b",
        r"\b(total|number|count)\b.*\b(dataset|procurement records)\b",
        r"\b(dataset|procurement dataset)\b.*\b(count|size|records?)\b",
    ]

    return any(
        re.search(pattern, q)
        for pattern in count_patterns
    )


def extract_year(question):
    match = re.search(
        r"\b((?:19|20)\d{2})\b",
        question,
    )

    return int(match.group(1)) if match else None


def is_year_question(question):
    q = normalize_text(question)

    year_phrases = [
        "which suppliers",
        "what suppliers",
        "suppliers received",
        "suppliers awarded",
        "contracts in",
        "awards in",
        "records in",
        "procurement in",
        "during",
        "for the year",
        "in the year",
    ]

    return (
        extract_year(question) is not None
        and any(phrase in q for phrase in year_phrases)
    )


def get_year_filtered_records(year):
    if YEAR_COLUMN is not None:
        years = pd.to_numeric(
            df[YEAR_COLUMN],
            errors="coerce",
        )

        return df[years == year].copy()

    if AWARD_DATE_COLUMN is not None:
        dates = pd.to_datetime(
            df[AWARD_DATE_COLUMN],
            errors="coerce",
        )

        return df[dates.dt.year == year].copy()

    raise ValueError(
        "Could not find an award year or award date column."
    )


# ============================================================
# PART 5: DIRECT CSV ANSWERS
# ============================================================

def answer_record_count():
    return (
        f"The dataset contains {len(df):,} procurement award records.",
        make_direct_source(
            "Count calculated directly from the cleaned CSV."
        ),
    )


def answer_year_query(year):
    records = get_year_filtered_records(year)

    if records.empty:
        return (
            f"No procurement award records were found for {year}.",
            make_direct_source(
                f"CSV filtered by award year {year}."
            ),
        )

    suppliers = (
        records[SUPPLIER_COLUMN]
        .dropna()
        .astype(str)
        .str.strip()
    )

    suppliers = sorted(
        {
            supplier
            for supplier in suppliers
            if supplier and supplier.lower() != "nan"
        },
        key=str.lower,
    )

    lines = [
        f"Procurement award records for {year}: {len(records):,}",
        f"Unique suppliers: {len(suppliers):,}",
        "",
        "Suppliers:",
    ]

    lines.extend(f"- {supplier}" for supplier in suppliers)

    return (
        "\n".join(lines),
        make_direct_source(
            f"CSV records filtered by award year {year}."
        ),
    )


def is_amount_question(question):
    q = normalize_text(question)

    amount_terms = [
        "amount",
        "how much",
        "contract value",
        "contract worth",
        "total value",
        "total award",
        "award value",
        "awarded amount",
        "contract award",
        "sum of",
    ]

    return any(term in q for term in amount_terms)


def is_supplier_amount_question(question):
    q = normalize_text(question)

    return (
        is_amount_question(q)
        and (
            "supplier" in q
            or "vendor" in q
            or "for " in q
            or "awarded to" in q
            or "received by" in q
        )
    )


def extract_supplier_candidate(question):
    """Extract the name following common supplier-question phrases."""

    text = str(question).strip()

    patterns = [
        r"\b(?:amount|value|award|contract)\b.*?\bfor\s+(.+?)\s*[?.!]*$",
        r"\b(?:supplier|vendor)\s+(.+?)\s*[?.!]*$",
        r"\bawarded\s+to\s+(.+?)\s*[?.!]*$",
        r"\breceived\s+by\s+(.+?)\s*[?.!]*$",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            candidate = match.group(1).strip()

            candidate = re.sub(
                r"[?.!,;:]+$",
                "",
                candidate,
            ).strip()

            candidate = re.sub(
                r"^(?:a|an|the)\s+",
                "",
                candidate,
                flags=re.IGNORECASE,
            ).strip()

            if candidate:
                return candidate

    return None


def answer_supplier_records(supplier_name):
    records = get_supplier_records(supplier_name)

    if records.empty:
        return (
            f"No matching supplier was found for '{supplier_name}' "
            "in the procurement dataset. No contract amount can be "
            "confirmed for that name. Check the spelling or provide "
            "the exact supplier name from the CSV.",
            make_direct_source(
                "Requested supplier checked against the CSV."
            ),
        )

    amounts = pd.to_numeric(
        records[AMOUNT_COLUMN],
        errors="coerce",
    )

    valid_amounts = amounts.dropna()
    total = valid_amounts.sum()

    lines = [
        f"Supplier: {supplier_name}",
        f"Number of award records: {len(records):,}",
        f"Records with valid amounts: {len(valid_amounts):,}",
        f"Total contract award amount: {format_amount(total)}",
        "",
        "Individual award records:",
    ]

    for number, (_, row) in enumerate(
        records.iterrows(),
        start=1,
    ):
        if AWARD_DATE_COLUMN is not None:
            date_value = row[AWARD_DATE_COLUMN]

            if pd.isna(date_value):
                date_value = "Date unavailable"
            else:
                date_value = str(date_value)
        else:
            date_value = "Date unavailable"

        lines.append(
            f"{number}. Date: {date_value} | "
            f"Amount: {format_amount(row[AMOUNT_COLUMN])}"
        )

    if len(valid_amounts) != len(records):
        lines.extend([
            "",
            "Note: Records with missing or invalid amounts "
            "were excluded from the total.",
        ])

    return (
        "\n".join(lines),
        make_direct_source(
            "Supplier records and amounts calculated directly "
            f"from the CSV for {supplier_name}."
        ),
    )


# ============================================================
# PART 6: ROBUST CHUNK TEXT AND SOURCE EXTRACTION
# ============================================================

def get_chunk_text(chunk):
    if isinstance(chunk, str):
        return chunk

    if not isinstance(chunk, dict):
        return str(chunk)

    for key in [
        "text",
        "content",
        "chunk",
        "page_content",
        "chunk_text",
    ]:
        value = chunk.get(key)

        if isinstance(value, str) and value.strip():
            return value

    nested = chunk.get("metadata")

    if isinstance(nested, dict):
        for key in ["text", "content", "chunk", "page_content"]:
            value = nested.get(key)

            if isinstance(value, str) and value.strip():
                return value

    return ""


def get_chunk_metadata(chunk):
    if not isinstance(chunk, dict):
        return {}

    nested = chunk.get("metadata")

    return nested if isinstance(nested, dict) else {}


def find_source_value(obj):
    """
    Search common source-name fields in the chunk and its
    nested metadata. Return a useful filename when available.
    """

    source_keys = [
        "source",
        "source_file",
        "source_filename",
        "filename",
        "file_name",
        "file",
        "document_name",
        "document",
        "document_path",
        "file_path",
        "path",
        "pdf_name",
    ]

    if not isinstance(obj, dict):
        return None

    # Prefer explicit source-related fields.
    for key in source_keys:
        value = obj.get(key)

        if isinstance(value, str) and value.strip():
            cleaned = value.strip()

            if cleaned.lower() not in {
                "unknown",
                "none",
                "null",
                "nan",
            }:
                return cleaned

    # Check nested metadata and other nested dictionaries.
    for value in obj.values():
        if isinstance(value, dict):
            result = find_source_value(value)

            if result:
                return result

    return None


def get_source_label(chunk, number=None):
    source = find_source_value(chunk)

    if source:
        # Normalize Windows paths and display the filename.
        source_normalized = source.replace("\\", "/")
        filename = source_normalized.rsplit("/", 1)[-1]

        if filename:
            return filename

    if number is not None:
        return f"Retrieved document chunk {number}"

    return "Retrieved document chunk"


def get_chunk_source_type(chunk):
    if isinstance(chunk, dict):
        explicit_type = chunk.get("source_type")

        if not explicit_type:
            explicit_type = get_chunk_metadata(chunk).get(
                "source_type"
            )

        if explicit_type:
            return str(explicit_type).lower()

    source = (find_source_value(chunk) or "").lower()

    if source.endswith(".csv"):
        return "csv"

    if source.endswith(".pdf") or "_ocr" in source:
        return "pdf"

    # OCR-generated text chunks may not retain their original
    # PDF filename. Inspect the available text as a fallback.
    text = get_chunk_text(chunk).lower()

    if "purchase order general terms" in text:
        return "pdf"

    if "vendor code of conduct" in text:
        return "pdf"

    return "document"


# ============================================================
# PART 7: FAISS RETRIEVAL
# ============================================================

def retrieve_documents(question):
    question_vector = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    question_vector = np.asarray(
        question_vector,
        dtype=np.float32,
    )

    search_count = min(TOP_K, index.ntotal)

    if search_count == 0:
        return []

    distances, indices = index.search(
        question_vector,
        search_count,
    )

    results = []

    for score, chunk_index in zip(
        distances[0],
        indices[0],
    ):
        if chunk_index < 0 or chunk_index >= len(metadata):
            continue

        chunk = metadata[chunk_index]

        results.append({
            "text": get_chunk_text(chunk),
            "metadata": get_chunk_metadata(chunk),
            "source": get_source_label(
                chunk,
                number=int(chunk_index) + 1,
            ),
            "source_type": get_chunk_source_type(chunk),
            "score": float(score),
        })

    return results


# ============================================================
# PART 8: GEMINI ANSWER GENERATION
# ============================================================

def generate_gemini_answer(question, results):
    if not results:
        return (
            "I could not find relevant information in "
            "the available procurement documents."
        )

    context_parts = []

    for number, result in enumerate(results, start=1):
        context_parts.append(
            f"[Source {number}: {result['source']}]\n"
            f"{result['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an AI-powered Supply Chain and Vendor
Documentation Assistant.

Answer the user's question using only the context below.

Rules:
1. Do not invent facts, requirements, dates, or amounts.
2. If the answer is not supported by the context, say so.
3. Cite the source filenames provided in the context.
4. For policy questions, explain the relevant requirements
   in clear bullet points.
5. Do not claim that a document says something unless
   the provided text supports it.
6. Keep the answer clear and relevant.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    last_error = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
            )

            answer = getattr(response, "text", None)

            if answer and answer.strip():
                return answer.strip()

            return (
                "Gemini did not return a text answer. "
                "Please try again."
            )

        except Exception as error:
            last_error = error
            error_message = str(error).lower()

            retryable = any(
                term in error_message
                for term in [
                    "503",
                    "unavailable",
                    "429",
                    "resource exhausted",
                    "rate limit",
                    "temporarily",
                    "timeout",
                ]
            )

            if retryable and attempt < 2:
                time.sleep(2 ** attempt)
                continue

            break

    print(f"Gemini error: {last_error}")

    return (
        "I could not generate an answer from Gemini right now. "
        "Please try again. Your local CSV and FAISS data "
        "are still available."
    )


# ============================================================
# PART 9: QUESTION ROUTER
# ============================================================

def answer_question(question):
    question = str(question).strip()

    if not question:
        return "Please enter a question.", []

    q = normalize_text(question)

    # STEP 1: Exact count from the CSV.
    if is_count_question(question):
        return answer_record_count()

    # STEP 2: Year-specific questions from the CSV.
    if is_year_question(question):
        year = extract_year(question)
        return answer_year_query(year)

    # STEP 3: Supplier amount questions.
    if is_supplier_amount_question(question):

        # Match an existing supplier first.
        named_supplier = find_named_supplier(question)

        if named_supplier:
            return answer_supplier_records(named_supplier)

        # If a supplier name was specified but does not exist,
        # return a direct no-match response. Do not ask Gemini
        # to guess or invent a contract amount.
        candidate = extract_supplier_candidate(question)

        if candidate:
            return (
                f"No matching supplier was found for '{candidate}' "
                "in the procurement dataset. No contract amount can "
                "be confirmed for that name. Check the spelling or "
                "provide the exact supplier name from the CSV.",
                make_direct_source(
                    "Requested supplier checked against the CSV."
                ),
            )

        return (
            "Please provide the supplier's name so I can check "
            "its contract award records and amount.",
            make_direct_source(
                "A supplier name is required for this lookup."
            ),
        )

    # STEP 4: Other questions use FAISS and Gemini.
    results = retrieve_documents(question)

    if not results:
        return (
            "I could not find relevant information in "
            "the available documents.",
            [],
        )

    answer = generate_gemini_answer(
        question,
        results,
    )

    sources = []

    for result in results:
        source_entry = {
            "source": result["source"],
            "type": result["source_type"],
        }

        if source_entry not in sources:
            sources.append(source_entry)

    return answer, sources


# ============================================================
# PART 10: COMMAND-LINE INTERFACE
# ============================================================

def main():
    print("=" * 65)
    print("AI-POWERED SUPPLY CHAIN & VENDOR DOCUMENTATION ASSISTANT")
    print("=" * 65)

    print("Ask questions about:")
    print("- Procurement record counts")
    print("- Suppliers and contract award amounts")
    print("- Contracts awarded in a particular year")
    print("- Vendor Code of Conduct")
    print("- World Bank purchase order terms and conditions")
    print("\nType 'exit' or 'quit' to stop.")
    print("-" * 65)

    while True:
        try:
            question = input("\nEnter your question: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting the assistant.")
            break

        if question.lower() in ["exit", "quit"]:
            print("Thank you for using the assistant.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer, sources = answer_question(question)

            print("\n" + "=" * 65)
            print("ANSWER")
            print("=" * 65)
            print(answer)

            if sources:
                print("\n" + "-" * 65)
                print("SOURCES")
                print("-" * 65)

                for source in sources:
                    print(
                        f"- {source['source']} "
                        f"({source['type']})"
                    )

                    description = source.get("description")

                    if description:
                        print(f"  {description}")

            print("=" * 65)

        except Exception as error:
            print("\nAn error occurred while answering the question.")
            print(f"Details: {error}")
            print(
                "Check that your CSV, FAISS index, metadata, "
                "embedding model, and .env file are available."
            )


if __name__ == "__main__":
    main()

