import io
import os
from typing import List, Tuple

import streamlit as st
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

from PyPDF2 import PdfReader
import docx

# -----------------------------
# Config
# -----------------------------
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "documents"
EMB_DIM = 768  # change if your collection uses different dimension
TOP_K = 5

# -----------------------------
# Milvus & embedding setup
# -----------------------------

@st.cache_resource
def get_milvus_collection():
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )
    return Collection(COLLECTION_NAME)

@st.cache_resource
def get_embedder():
    # Use any embedding model you prefer (must match your collection embeddings)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model

def embed_texts(texts: List[str]):
    model = get_embedder()
    return model.encode(texts, convert_to_numpy=True).tolist()

# -----------------------------
# Document text extraction
# -----------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text.append(page_text)
    return "\n".join(text)

def extract_text_from_docx(file_bytes: bytes) -> str:
    f = io.BytesIO(file_bytes)
    document = docx.Document(f)
    paragraphs = [p.text for p in document.paragraphs]
    return "\n".join(paragraphs)

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def extract_text(file) -> Tuple[str, str]:
    """
    Returns (text, file_type)
    file: UploadedFile from streamlit
    """
    name = file.name.lower()
    data = file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data), "pdf"
    elif name.endswith(".docx"):
        return extract_text_from_docx(data), "docx"
    elif name.endswith(".txt"):
        return extract_text_from_txt(data), "txt"
    else:
        # fallback – treat as text
        return extract_text_from_txt(data), "txt"

# -----------------------------
# Milvus operations
# -----------------------------

def search_milvus(query: str, top_k: int = TOP_K):
    col = get_milvus_collection()
    # load collection into memory
    col.load()

    query_vec = embed_texts([query])
    # Adjust anns_field, metric_type, and params according to your index
    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 10},
    }

    res = col.search(
        data=query_vec,
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["id", "content", "doc_name"],
    )
    # res is List[List[Hit]]
    return res[0]

def insert_document(text: str, doc_name: str):
    """
    Insert a single document as one record.
    For production, you may want to chunk text and insert multiple rows.
    """
    if not text.strip():
        return

    col = get_milvus_collection()
    col.load()

    embedding = embed_texts([text])[0]

    # Simple auto-increment-like id (not safe for concurrency; adapt to your needs)
    # In a real setup, define id on client side or let Milvus use auto_id.
    # Here, assume id is auto_id=True in schema -> then don't pass id field.
    # Example below assumes manual id for clarity; change according to your schema.
    # Retrieve current entity count:
    col.release()
    col.load()
    count = col.num_entities
    new_id = count + 1

    mr = col.insert([
        [new_id],          # id
        [text],            # content
        [doc_name],        # doc_name
        [embedding],       # vector
    ])
    col.flush()
    return mr

# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="Semantic Search over Docs", layout="wide")
    st.title("Semantic Search over Scanned Docs / PDFs / DOCX / Texts (Milvus)")

    st.sidebar.header("Upload & Ingest")
    uploaded_files = st.sidebar.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.sidebar.button("Ingest uploaded files into Milvus") and uploaded_files:
        with st.spinner("Extracting text and inserting into Milvus..."):
            for f in uploaded_files:
                f.seek(0)
                text, ftype = extract_text(f)
                if not text.strip():
                    st.warning(f"{f.name}: no text extracted (maybe scanned-only PDF).")
                    continue
                insert_document(text, f.name)
            st.success("Ingestion complete.")

    st.markdown("---")
    st.subheader("Semantic Search")

    query = st.text_input("Enter your search query")
    top_k = st.slider("Top K results", 1, 20, TOP_K)

    if st.button("Search") and query.strip():
        with st.spinner("Searching..."):
            hits = search_milvus(query, top_k=top_k)

        st.write(f"Found {len(hits)} results:")

        for i, hit in enumerate(hits, start=1):
            score = hit.distance
            doc_name = hit.entity.get("doc_name", "N/A")
            content = hit.entity.get("content", "")

            with st.expander(f"{i}. {doc_name} (score={score:.4f})"):
                st.write(content[:2000] + ("..." if len(content) > 2000 else ""))

    st.info(
        "For scanned PDFs (image-only), integrate an OCR pipeline (e.g., Tesseract, "
        "Amazon Textract, Azure OCR, or PaddleOCR) in `extract_text_from_pdf` before ingest."
    )

if __name__ == "__main__":
    main()
