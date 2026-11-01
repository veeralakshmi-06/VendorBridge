
import streamlit as st
from src import rag_pipeline
from contextlib import redirect_stdout
from io import StringIO
import re


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Supply Chain & Vendor Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM UI STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 1.5rem 1.7rem;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            background: linear-gradient(120deg, #f8fbff, #eef5ff);
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            color: #17365d;
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .hero p {
            color: #52647a;
            font-size: 1rem;
            margin-bottom: 0;
        }

        .section-label {
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        [data-testid="stChatMessage"] {
            border: 1px solid #e8edf4;
            border-radius: 14px;
            padding: 1rem;
            margin-bottom: 0.8rem;
        }

        [data-testid="stChatInput"] {
            border-radius: 14px;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid #e5eaf1;
        }

        div[data-testid="stExpander"] {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
        }

        .stButton button {
            border-radius: 10px;
        }

        .footer-note {
            color: #8491a3;
            font-size: 0.82rem;
            text-align: center;
            padding-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_answer(text):
    """Clean escaped formatting while preserving the answer."""
    text = str(text).strip()

    # Handle literal escaped newlines if present.
    text = text.replace("\\r\\n", "\n")
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", "    ")

    # Restore escaped Markdown characters.
    text = re.sub(r"\\([*#_`])", r"\1", text)

    # Remove surrounding whitespace.
    return text.strip()


def parse_pipeline_result(result, captured_text):
    """
    Extract the answer and source list from the existing RAG pipeline.
    Supports (answer, sources), plain strings, and printed output.
    """
    answer = ""
    sources = []

    if isinstance(result, (tuple, list)) and len(result) >= 1:
        answer = result[0] if isinstance(result[0], str) else str(result[0])

        if len(result) >= 2 and isinstance(result[1], list):
            sources = result[1]

    elif isinstance(result, str):
        answer = result

    elif result is not None:
        answer = str(result)

    if not answer.strip():
        answer = captured_text.strip()

    # Handle older pipeline output that prints a tuple instead
    # of returning the answer and sources.
    if not sources and answer.startswith("("):
        try:
            import ast

            parsed = ast.literal_eval(answer)

            if isinstance(parsed, (tuple, list)) and len(parsed) >= 1:
                answer = (
                    parsed[0]
                    if isinstance(parsed[0], str)
                    else str(parsed[0])
                )

                if len(parsed) >= 2 and isinstance(parsed[1], list):
                    sources = parsed[1]

        except (ValueError, SyntaxError):
            pass

    answer = clean_answer(answer)

    if not answer:
        answer = (
            "I couldn't generate an answer for that question. "
            "Please try asking it another way."
        )

    return answer, sources


def display_sources(sources):
    """Display source documents in a readable expandable section."""
    if not isinstance(sources, list) or not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        seen = set()

        for source in sources:
            if isinstance(source, dict):
                filename = (
                    source.get("source")
                    or source.get("filename")
                    or source.get("file")
                    or "Source document"
                )

                description = source.get("description", "")
                source_type = source.get("type", "")

                identity = (filename, description)

                if identity in seen:
                    continue

                seen.add(identity)

                st.markdown(f"**📄 {filename}**")

                if source_type:
                    st.caption(f"Type: {source_type.upper()}")

                if description:
                    st.write(description)

                st.divider()

            else:
                st.write(str(source))


def display_message(message):
    """Render a saved chat message with its source documents."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(message.get("sources", []))


def process_question(question):
    """Run the existing RAG pipeline without modifying it."""
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and preparing your answer..."):
            captured_output = StringIO()

            try:
                with redirect_stdout(captured_output):
                    result = rag_pipeline.answer_question(question)

                answer, sources = parse_pipeline_result(
                    result,
                    captured_output.getvalue(),
                )

            except Exception:
                st.error(
                    "An error occurred while processing your question. "
                    "Please check the VS Code terminal for details."
                )
                return

        st.markdown(answer)
        display_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 📄 Supply Chain AI")

    st.caption("Procurement and vendor document assistant")

    st.divider()

    st.markdown("### About")
    st.write(
        "Ask questions about procurement awards, suppliers, "
        "vendor ethics, and purchase order terms. Answers are "
        "generated using your existing RAG pipeline."
    )

    st.markdown("### Available topics")

    st.markdown(
        """
        - 📊 Procurement record counts
        - 🏢 Supplier award amounts
        - 📅 Award records by year
        - ⚖️ Vendor Code of Conduct
        - 📑 Purchase order terms
        """
    )

    st.divider()

    st.markdown("### Conversation")

    st.caption(
        f"{len(st.session_state.messages) // 2} "
        "question-and-answer messages may be present in this session."
    )

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📄 Supply Chain & Vendor Documentation Assistant</h1>
        <p>
            Your AI assistant for procurement data, supplier contracts,
            vendor policies, and purchase order documentation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# WELCOME MESSAGE AND SUGGESTED QUESTIONS
# ============================================================

if not st.session_state.messages:
    st.markdown("### 👋 How can I help you today?")

    st.write(
        "Ask a question in your own words. I'll search the available "
        "procurement records and vendor documents for relevant information."
    )

    st.markdown(
        '<div class="section-label">TRY ASKING</div>',
        unsafe_allow_html=True,
    )

    suggestions = [
        "How many procurement records are available?",
        "What is the contract award amount for FRED Engineering S.r.l.?",
        "What ethical requirements must vendors follow?",
        "What are the purchase order terms and conditions?",
    ]

    for start in range(0, len(suggestions), 2):
        col1, col2 = st.columns(2)

        for col, index in zip(
            (col1, col2),
            range(start, min(start + 2, len(suggestions))),
        ):
            with col:
                if st.button(
                    suggestions[index],
                    key=f"suggestion_{index}",
                    use_container_width=True,
                ):
                    st.session_state.pending_question = suggestions[index]
                    st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

for saved_message in st.session_state.messages:
    display_message(saved_message)


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "💬 Ask about procurement, suppliers, or vendor documents..."
)

# Suggested questions are processed through the same chat pipeline.
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question and question.strip():
    process_question(question.strip())


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer-note">
        Powered by FAISS · Sentence Transformers · Gemini · Streamlit
        <br>
        Verify important procurement and contractual information
        against the original documents.
    </div>
    """,
    unsafe_allow_html=True,
)