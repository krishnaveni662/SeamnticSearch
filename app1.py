import os, io, tempfile, base64
from dotenv import load_dotenv
import requests
from huggingface_hub import InferenceClient
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import time
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import camelot

load_dotenv()
HF = os.environ["HF_API_TOKEN"]
# Text/tables embedder (HF Inference API, CPU-friendly)
EMBED_MODEL_TEXT = os.getenv("EMBED_MODEL_TEXT", "BAAI/bge-base-en-v1.5")
# Image embedder (CLIP)
EMBED_MODEL_IMAGE = os.getenv("EMBED_MODEL_IMAGE", "openai/clip-vit-base-patch32")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Qwen2.5-VL-7B-Instruct")
HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
PORT = os.getenv("MILVUS_PORT", "19530")

# Initialize Inference Clients
text_client = InferenceClient(model=EMBED_MODEL_TEXT, token=HF)
image_client = InferenceClient(model=EMBED_MODEL_IMAGE, token=HF)
rerank_client = InferenceClient(model=RERANK_MODEL, token=HF)

# --- RBAC setup ---
ROLES = {
    "admin": {"classification": ["public", "internal", "secret"]},
    "employee": {"classification": ["public", "internal"]},
    "guest": {"classification": ["public"]},
}

# --- Milvus setup ---
COLLECTION = "docs"
DIM = 768  # bge-base outputs 768 dims; CLIP is padded to 768

def ensure_collection():
    connections.connect(alias="default", host=HOST, port=PORT)
    
    # Define schema fields
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="meta_classification", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="meta_source", dtype=DataType.VARCHAR, max_length=256),
    ]
    schema = CollectionSchema(fields, description="docs with RBAC")
    
    # Check if collection exists
    if utility.has_collection(COLLECTION):
        try:
            coll = Collection(COLLECTION)
            # Check if schema matches (has all required fields)
            schema_fields = [f.name for f in coll.schema.fields]
            required_fields = ["embedding", "text", "modality", "meta_classification", "meta_source"]
            if all(field in schema_fields for field in required_fields):
                # Check if index exists, create if not
                if not coll.has_index():
                    coll.create_index("embedding", {"index_type":"IVF_FLAT","metric_type":"IP","params":{"nlist":1024}})
                coll.load()
                return coll
            else:
                # Drop and recreate if schema doesn't match
                utility.drop_collection(COLLECTION)
                time.sleep(1)
        except Exception as e:
            # If there's an error accessing collection, drop and recreate
            try:
                utility.drop_collection(COLLECTION)
                time.sleep(1)
            except:
                pass
    
    # Create new collection - use utility.create_collection for explicit creation
    try:
        # Try to create collection explicitly
        if not utility.has_collection(COLLECTION):
            coll = Collection(COLLECTION, schema, consistency_level="Strong")
        else:
            coll = Collection(COLLECTION)
    except Exception as e:
        # If creation fails, try alternative method
        coll = Collection(COLLECTION, schema, consistency_level="Strong")
    
    # Create index if it doesn't exist
    try:
        if not coll.has_index():
            coll.create_index("embedding", {"index_type":"IVF_FLAT","metric_type":"IP","params":{"nlist":1024}})
    except Exception:
        # Index might already exist or collection needs to be loaded first
        pass
    
    coll.load()
    return coll

# --- HF helpers ---
def hf_embed_text(texts):
    """Embed text/tables with BGE; returns list of vectors (each vector is a flat list of floats)."""
    # Handle single text or list of texts
    if isinstance(texts, str):
        texts = [texts]
    
    def normalize_embedding(emb):
        """Normalize embedding to flat list of floats."""
        # Handle numpy arrays
        try:
            import numpy as np
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
        except ImportError:
            pass
        
        if isinstance(emb, list):
            # Unwrap nested lists
            while isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                emb = emb[0]
            # Ensure all are floats
            return [float(x) for x in emb]
        return emb
    
    try:
        # Use InferenceClient which handles endpoint details
        data = text_client.feature_extraction(texts)
        # Normalize the response
        if isinstance(data, list):
            if len(data) > 0:
                if isinstance(data[0], list):
                    # Multiple embeddings or nested structure
                    return [normalize_embedding(emb) for emb in data]
                else:
                    # Single embedding, wrap it
                    return [normalize_embedding(data)]
        return [normalize_embedding(data)]
    except Exception as e:
        # Fallback to direct API call if InferenceClient fails
        url = f"https://api-inference.huggingface.co/models/{EMBED_MODEL_TEXT}"
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {HF}"},
            json={"inputs": texts, "options":{"wait_for_model":True}},
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        # Normalize the response
        if isinstance(data, list):
            if len(data) > 0:
                if isinstance(data[0], list):
                    return [normalize_embedding(emb) for emb in data]
                else:
                    return [normalize_embedding(data)]
        return [normalize_embedding(data)]

def hf_embed_clip_image(image_bytes):
    """Embed image bytes with CLIP; pads to DIM to align with collection. Returns flat list of floats."""
    def normalize_vec(v):
        """Normalize vector to flat list of floats."""
        # Handle numpy arrays
        try:
            import numpy as np
            if isinstance(v, np.ndarray):
                v = v.tolist()
        except ImportError:
            pass
        
        if isinstance(v, dict) and "image_embeds" in v:
            v = v["image_embeds"]
        elif isinstance(v, list):
            # Unwrap nested lists
            while isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                v = v[0]
        # Ensure all are floats
        return [float(x) for x in v] if isinstance(v, list) else v
    
    try:
        # Use InferenceClient for image embedding
        vec = image_client.feature_extraction(image=image_bytes)
        vec = normalize_vec(vec)
    except Exception as e:
        # Fallback to direct API call
        url = f"https://api-inference.huggingface.co/models/{EMBED_MODEL_IMAGE}"
        img_b64 = base64.b64encode(image_bytes).decode('utf-8')
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {HF}"},
            json={"inputs": {"image": img_b64}, "options":{"wait_for_model":True}},
            timeout=60
        )
        resp.raise_for_status()
        vec = resp.json()
        vec = normalize_vec(vec)
    
    # pad CLIP's 512-dim to 768
    if len(vec) < DIM:
        vec = vec + [0.0]*(DIM - len(vec))
    return vec[:DIM]

def hf_embed_clip_text(text):
    """Optional: text query to search image embeddings. Returns flat list of floats."""
    def normalize_vec(v):
        """Normalize vector to flat list of floats."""
        # Handle numpy arrays
        try:
            import numpy as np
            if isinstance(v, np.ndarray):
                v = v.tolist()
        except ImportError:
            pass
        
        if isinstance(v, dict) and "text_embeds" in v:
            v = v["text_embeds"]
        elif isinstance(v, list):
            # Unwrap nested lists
            while isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                v = v[0]
        # Ensure all are floats
        return [float(x) for x in v] if isinstance(v, list) else v
    
    try:
        # Use InferenceClient for text embedding
        vec = image_client.feature_extraction(text=text)
        vec = normalize_vec(vec)
    except Exception as e:
        # Fallback to direct API call
        url = f"https://api-inference.huggingface.co/models/{EMBED_MODEL_IMAGE}"
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {HF}"},
            json={"inputs": {"text": text}, "options":{"wait_for_model":True}},
            timeout=60
        )
        resp.raise_for_status()
        vec = resp.json()
        vec = normalize_vec(vec)
    
    if len(vec) < DIM:
        vec = vec + [0.0]*(DIM - len(vec))
    return vec[:DIM]

def hf_rerank(query, docs):
    # Use direct API call for reranking (InferenceClient doesn't have rerank method)
    url = f"https://api-inference.huggingface.co/models/{RERANK_MODEL}"
    payload = {"inputs": {"query": query, "documents": docs}, "options":{"wait_for_model":True}}
    resp = requests.post(url, headers={"Authorization": f"Bearer {HF}"}, json=payload, timeout=60)
    resp.raise_for_status()
    scores = resp.json()
    # returns list of {"index": i, "score": s} or {"score": s} per doc
    if isinstance(scores, list) and len(scores) > 0:
        # Ensure format is consistent
        if not isinstance(scores[0], dict) or "score" not in scores[0]:
            # If format is different, normalize it
            scores = [{"index": i, "score": float(s)} for i, s in enumerate(scores)]
    return scores

# --- Extraction helpers ---
def extract_text_from_pdf(file_bytes):
    """Text extraction via PyMuPDF; poppler-free."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    texts = []
    for page in doc:
        txt = page.get_text("text") or ""
        texts.append(txt.strip())
    doc.close()
    return "\n".join(texts)

def _page_pix_to_pil(page, zoom=2):
    """Render a PDF page to PIL.Image for OCR."""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return img

def extract_text_from_pdf_ocr(file_bytes):
    """Fallback OCR using PyMuPDF render + Tesseract."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    chunks = []
    for page in doc:
        img = _page_pix_to_pil(page, zoom=2)
        txt = pytesseract.image_to_string(img)
        chunks.append(txt)
    doc.close()
    return "\n".join(chunks)

def extract_text_from_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(img)

def extract_tables_from_pdf(file_bytes):
    # crude table extractor; optional
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tables = camelot.read_pdf(tmp.name, pages="all")
    os.unlink(tmp.name)
    return "\n".join([t.df.to_string() for t in tables]) if len(tables) else ""

# --- Text Chunking ---
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into semantically meaningful chunks with overlap for better context preservation.
    Uses larger chunks (1000 chars) to preserve semantic meaning and breaks at natural boundaries.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum characters per chunk (default 1000 for better semantic preservation)
        chunk_overlap: Number of characters to overlap between chunks (default 200 for context)
    
    Returns:
        List of text chunks
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # If text is small enough, return as single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        
        # If this isn't the last chunk, try to break at a semantic boundary
        if end < text_len:
            # Look for good break points within the last 300 chars (larger window for better breaks)
            search_start = max(start, end - 300)
            
            # Priority order for semantic breaks (most semantic first):
            # 1. Paragraph breaks (best semantic boundary)
            # 2. Sentence endings with punctuation
            # 3. Sentence endings
            # 4. Line breaks
            # 5. Whitespace (last resort)
            break_points = [
                '\n\n\n',    # Multiple paragraph breaks (highest priority)
                '\n\n',      # Paragraph break
                '.\n\n',     # Sentence end + paragraph
                '!\n\n',     # Exclamation + paragraph
                '?\n\n',     # Question + paragraph
                '.\n',       # Sentence with newline
                '!\n',       # Exclamation with newline
                '?\n',       # Question with newline
                '. ',        # Sentence end with space
                '! ',        # Exclamation with space
                '? ',        # Question with space
                '.\t',       # Sentence with tab
                '\n',        # Single newline
                '. ',        # Period with space (fallback)
                '; ',        # Semicolon (clause boundary)
                ', ',        # Comma (phrase boundary, last resort)
            ]
            
            best_break = end
            best_priority = len(break_points)  # Lower number = higher priority
            
            for i, break_point in enumerate(break_points):
                # Find the last occurrence of this break point before end
                pos = text.rfind(break_point, search_start, end)
                if pos != -1 and pos > search_start:
                    # Found a break point, use it if it's better priority
                    if i < best_priority:
                        best_break = pos + len(break_point)
                        best_priority = i
            
            # Use the best break point found
            if best_break > search_start:
                end = best_break
        
        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move start position with overlap (but don't go backwards or repeat)
        if end < text_len:
            # Calculate overlap start, ensuring we don't go backwards
            overlap_start = end - chunk_overlap
            start = max(start + 1, overlap_start)
        else:
            break
        
        # Safety check to avoid infinite loops
        if start >= text_len:
            break
    
    return chunks

# --- Ingest ---
def ingest(file, classification, source):
    coll = ensure_collection()
    bytes_data = file.read()
    text = ""
    modality = "text"
    if file.type == "application/pdf":
        text = extract_text_from_pdf(bytes_data)
        if len(text.strip()) < 10:
            text = extract_text_from_pdf_ocr(bytes_data)
        text += "\n" + extract_tables_from_pdf(bytes_data)
    elif file.type.startswith("image/"):
        modality = "image"
        text = extract_text_from_image(bytes_data)
    else:
        text = bytes_data.decode("utf-8", errors="ignore")
    text = text.strip()
    
    # Handle images separately (no chunking needed)
    if modality == "image":
        emb = hf_embed_clip_image(bytes_data)
        
        # Normalize embedding
        try:
            import numpy as np
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
        except ImportError:
            pass
        
        if isinstance(emb, list):
            while isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                emb = emb[0]
            emb = [float(x) for x in emb]
        else:
            return False, f"Invalid embedding format: {type(emb).__name__}"
        
        # Ensure embedding has correct dimension
        if len(emb) != DIM:
            if len(emb) < DIM:
                emb = emb + [0.0] * (DIM - len(emb))
            else:
                emb = emb[:DIM]
        
        # Truncate text if needed (for OCR results)
        MAX_TEXT_LENGTH = 4096
        text_chunk = text[:MAX_TEXT_LENGTH] if len(text) > MAX_TEXT_LENGTH else text
        
        try:
            coll.insert([[emb], [text_chunk], [modality], [classification], [source]])
            coll.flush()
            coll.load()
            return True, f"Ingested 1 image chunk from {source}"
        except Exception as e:
            error_msg = str(e).replace('<', '&lt;').replace('>', '&gt;')
            return False, f"Insert failed: {error_msg}"
    
    # Handle text documents with chunking
    if not text:
        return False, "No text extracted"
    
    # Chunk the text (1000 chars per chunk with 200 char overlap for better semantic meaning)
    text_chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
    
    if not text_chunks:
        return False, "No text chunks created"
    
    # Embed chunks in batches to ensure we get one embedding per chunk
    # Some APIs may concatenate multiple texts, so we'll process individually or in small batches
    chunk_embeddings = []
    BATCH_SIZE = 10  # Process 10 chunks at a time to balance efficiency and reliability
    
    try:
        for i in range(0, len(text_chunks), BATCH_SIZE):
            batch = text_chunks[i:i+BATCH_SIZE]
            batch_embeddings = hf_embed_text(batch)
            
            # Handle different response formats
            if isinstance(batch_embeddings, list):
                # Check if we got the right number of embeddings
                if len(batch_embeddings) == len(batch):
                    chunk_embeddings.extend(batch_embeddings)
                elif len(batch_embeddings) == 1 and len(batch) > 1:
                    # API returned single embedding for batch - process individually
                    for chunk in batch:
                        single_emb = hf_embed_text([chunk])
                        if isinstance(single_emb, list) and len(single_emb) > 0:
                            chunk_embeddings.append(single_emb[0])
                        else:
                            return False, f"Failed to get embedding for chunk {i}"
                else:
                    return False, f"Unexpected embedding count: {len(batch_embeddings)} for {len(batch)} chunks"
            else:
                return False, f"Invalid embedding response format: {type(batch_embeddings)}"
    except Exception as e:
        return False, f"Embedding failed: {str(e)}"
    
    if len(chunk_embeddings) != len(text_chunks):
        return False, f"Embedding count mismatch: {len(chunk_embeddings)} embeddings for {len(text_chunks)} chunks"
    
    # Prepare data for batch insert
    embeddings_list = []
    texts_list = []
    modalities_list = []
    classifications_list = []
    sources_list = []
    
    MAX_TEXT_LENGTH = 4096
    
    for chunk, emb in zip(text_chunks, chunk_embeddings):
        # Normalize embedding
        try:
            import numpy as np
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
        except ImportError:
            pass
        
        if isinstance(emb, list):
            while isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                emb = emb[0]
            emb = [float(x) for x in emb]
        else:
            continue  # Skip invalid embeddings
        
        # Ensure embedding has correct dimension
        if len(emb) != DIM:
            if len(emb) < DIM:
                emb = emb + [0.0] * (DIM - len(emb))
            else:
                emb = emb[:DIM]
        
        # Ensure chunk text fits within limit (should already be small, but double-check)
        chunk_text_final = chunk[:MAX_TEXT_LENGTH] if len(chunk) > MAX_TEXT_LENGTH else chunk
        
        embeddings_list.append(emb)
        texts_list.append(chunk_text_final)
        modalities_list.append(modality)
        classifications_list.append(classification)
        sources_list.append(source)
    
    if not embeddings_list:
        return False, "No valid embeddings created"
    
    # Batch insert all chunks
    try:
        coll.insert([embeddings_list, texts_list, modalities_list, classifications_list, sources_list])
        coll.flush()
        coll.load()
        return True, f"Ingested {len(embeddings_list)} chunk(s) from {source} (original text: {len(text)} chars)"
    except Exception as e:
        error_msg = str(e).replace('<', '&lt;').replace('>', '&gt;')
        return False, f"Insert failed: {error_msg}"

# --- Search ---
def search(query, role, top_k=5):
    coll = ensure_collection()
    allowed = ROLES.get(role, {}).get("classification", [])
    expr = f'meta_classification in {allowed}' if allowed else ""
    # search text/table embeddings
    q_emb_text = hf_embed_text([query])[0]
    res_text = coll.search(
        data=[q_emb_text], anns_field="embedding",
        param={"metric_type":"IP","params":{"nprobe":16}},
        limit=top_k*3, expr=f"({expr}) and modality == 'text'" if expr else "modality == 'text'",
        output_fields=["text","meta_source","meta_classification","modality"]
    )
    # search images using CLIP text encoder
    q_emb_clip = hf_embed_clip_text(query)
    res_img = coll.search(
        data=[q_emb_clip], anns_field="embedding",
        param={"metric_type":"IP","params":{"nprobe":16}},
        limit=top_k*3, expr=f"({expr}) and modality == 'image'" if expr else "modality == 'image'",
        output_fields=["text","meta_source","meta_classification","modality"]
    )
    hits = list(res_text[0]) + list(res_img[0])
    if not hits:
        return []
    texts = [h.entity.get("text") for h in hits]
    rerank = hf_rerank(query, texts)
    reranked = sorted(zip(hits, rerank), key=lambda x: x[1]["score"], reverse=True)[:top_k]
    return [{
        "score": float(r[1]["score"]),
        "text": r[0].entity.get("text"),
        "source": r[0].entity.get("meta_source"),
        "classification": r[0].entity.get("meta_classification"),
        "modality": r[0].entity.get("modality"),
    } for r in reranked]

# --- UI ---
st.set_page_config(page_title="Semantic Search + RBAC", layout="wide")
st.title("Semantic Search with Milvus, HF, Qwen2.5-VL-7B rerank")

tab_ingest, tab_search = st.tabs(["Ingest", "Search"])

with tab_ingest:
    role = st.selectbox("Role (for writing metadata only)", list(ROLES.keys()))
    classification = st.selectbox("Classification", ROLES[role]["classification"])
    file = st.file_uploader("Upload document/image/pdf", type=["pdf","txt","md","png","jpg","jpeg"])
    source = st.text_input("Source label", value="upload")
    if st.button("Ingest") and file:
        ok, msg = ingest(file, classification, source)
        # Ensure message is a plain string for Streamlit (no HTML, no objects)
        if msg:
            # Convert to string and escape any problematic characters
            msg_str = str(msg)
            # Remove any object references that might cause Streamlit issues
            msg_str = msg_str.replace('_repr_html_', '').replace('()', '')
        else:
            msg_str = "Unknown error"
        
        if ok:
            st.success(msg_str)
        else:
            st.error(msg_str)

with tab_search:
    role = st.selectbox("Role", list(ROLES.keys()), key="search_role")
    query = st.text_input("Query")
    topk = st.slider("Top K", 1, 10, 5)
    if st.button("Search") and query:
        results = search(query, role, topk)
        for r in results:
            st.markdown(f"**Score:** {r['score']:.3f} | **Mod:** {r['modality']} | **Class:** {r['classification']} | **Src:** {r['source']}")
            st.write(r["text"][:1000] + ("..." if len(r["text"])>1000 else ""))