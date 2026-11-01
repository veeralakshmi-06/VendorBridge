# VendorBridge
### AI-Powered Supply Chain & Vendor Documentation Assistant

VendorBridge is an AI-powered assistant designed to simplify procurement data analysis and vendor document understanding. It combines Retrieval-Augmented Generation (RAG), semantic search, and Large Language Models (LLMs) to provide relevant answers from procurement records and vendor documentation.

## About the Project

Organizations maintain large amounts of procurement records, vendor policies, and contractual documents. Manually searching through these records can be time-consuming.

VendorBridge addresses this challenge by allowing users to ask questions in natural language and receive relevant, context-based answers through an interactive chat interface.

## Key Features

- **Procurement Data Analysis:** Retrieve supplier details, contract award amounts, and procurement record counts.
- **Vendor Document Understanding:** Ask questions about vendor policies and contractual requirements.
- **OCR-Based Text Extraction:** Extract text from scanned PDF documents using Tesseract OCR.
- **Document Chunking:** Divide extracted content into smaller, searchable chunks.
- **Semantic Search:** Retrieve relevant information using FAISS vector similarity search.
- **AI-Powered Responses:** Generate context-aware answers using Google Gemini.
- **Source Identification:** Display the documents used to generate answers.
- **Interactive Chat Interface:** User-friendly interface developed using Streamlit.
- **Direct Data Processing:** Handle supported numerical queries directly from procurement CSV records.

## System Architecture

The project follows a Retrieval-Augmented Generation workflow:

1. **Data Collection:** Collect procurement CSV records and vendor PDF documents.
2. **Preprocessing:** Clean procurement data and extract text from scanned PDFs using OCR.
3. **Chunking:** Divide document content into smaller chunks.
4. **Embedding Generation:** Convert text chunks into numerical vectors using Sentence Transformers.
5. **Vector Storage:** Store embeddings and metadata using FAISS.
6. **Retrieval:** Search for relevant chunks based on user queries.
7. **Generation:** Pass retrieved context to Google Gemini to generate relevant answers.
8. **Presentation:** Display answers and source documents through Streamlit.

*Note: Supported numerical CSV queries are also handled directly using Pandas instead of going through the complete RAG retrieval process.*

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive user interface |
| FAISS | Vector similarity search |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Embedding model |
| Google Gemini | AI response generation |
| Tesseract OCR | Text extraction from scanned PDFs |
| Pandas | Data cleaning and analysis |
| NumPy | Numerical operations |

## Dataset

The project uses:

- Procurement contract award records in CSV format.
- Vendor Code of Conduct PDF.
- World Bank Group Purchase Order General Terms and Conditions PDF.

The cleaned procurement dataset contains **4,799 procurement award records**.

## Project Structure

```text
VendorBridge/
│
├── app.py
├── src/
│   └── rag_pipeline.py
│
├── data/
│   └── Original datasets and PDF documents
│
├── processed_data/
│   ├── cleaned_procurement_data.csv
│   ├── Vendor Code of Conduct_ocr.txt
│   └── WBG Terms and Conditions_ocr.txt
│
├── chunks/
│   └── all_chunks.json
│
├── vector_store/
│   ├── faiss_index.bin
│   ├── chunks_metadata.json
│   └── embeddings.npy
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

*The structure is illustrative. Actual folders and filenames may differ.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/veeralakshmi-06/VendorBridge.git
cd VendorBridge
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Gemini API Key

Create a `.env` file in the project root directory:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder with your Gemini API key.

**Important:** Never upload your actual `.env` file or API key to GitHub.

### 4. Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your default web browser.

## Example Queries

Users can ask questions such as:

- How many procurement records are available?
- What is the contract award amount for FRED Engineering S.r.l.?
- What ethical requirements must vendors follow?
- What are the contractual requirements for vendors?

## Future Enhancements

- Support for additional document formats.
- Improved multilingual document processing.
- Advanced supplier analytics and visualizations.
- Enhanced document verification.
- Cloud deployment and scalability.

## Project Information

**Project Name:** VendorBridge

**Project Type:** Knowledge Engineering and RAG Mini-Project

**Domain:** Artificial Intelligence | Natural Language Processing | Information Retrieval

**Author:** Veeralakshmi

**GitHub:** [veeralakshmi-06](https://github.com/veeralakshmi-06)

---

*VendorBridge – Connecting Procurement Data with Intelligent Document Insights.*