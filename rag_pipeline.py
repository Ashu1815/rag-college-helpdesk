"""
Core RAG (Retrieval-Augmented Generation) pipeline.

Pipeline stages:
1. LOAD   - read .txt / .pdf files from data/sample_docs
2. CHUNK  - split documents into overlapping text chunks
3. EMBED  - convert chunks into vectors using a local sentence-transformer
4. STORE  - persist vectors in a local ChromaDB collection
5. RETRIEVE - given a user query, find the most relevant chunks
6. GENERATE - feed retrieved chunks + query into a local LLM (via Ollama)
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions
import ollama

# ---------- CONFIG ----------
DATA_DIR = "data/sample_docs"
DB_DIR = "vectorstore"
COLLECTION_NAME = "college_helpdesk"
EMBED_MODEL = "all-MiniLM-L6-v2"   # small, fast, runs locally on CPU
LLM_MODEL = "llama3.2:1b"             # change to any model you've pulled via `ollama pull <model>`
CHUNK_SIZE = 500                   # characters per chunk
CHUNK_OVERLAP = 80                 # overlap between consecutive chunks
TOP_K = 3                          # number of chunks to retrieve per query


def load_documents(data_dir: str = DATA_DIR):
    """Read all .txt files from the data directory."""
    docs = []
    for filepath in glob.glob(os.path.join(data_dir, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"source": os.path.basename(filepath), "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks so each chunk retains some context
    from its neighbor. This avoids cutting a fact in half between chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def get_chroma_collection():
    """Initialize (or load) a persistent local Chroma collection using a
    local sentence-transformer embedding function -- no API key required."""
    client = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    return collection


def build_index():
    """Load -> chunk -> embed -> store. Run this once (or whenever source
    documents change) via `python ingest.py`."""
    collection = get_chroma_collection()

    # Clear any existing entries so re-running ingest.py doesn't duplicate data
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    docs = load_documents()
    if not docs:
        print(f"No .txt files found in {DATA_DIR}. Add some documents first.")
        return

    all_chunks, all_ids, all_metadatas = [], [], []
    chunk_counter = 0
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_counter}")
            all_metadatas.append({"source": doc["source"]})
            chunk_counter += 1

    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadatas)
    print(f"Indexed {len(all_chunks)} chunks from {len(docs)} document(s) into '{COLLECTION_NAME}'.")


def retrieve(query: str, top_k: int = TOP_K):
    """Return the top_k most relevant chunks for a given query."""
    collection = get_chroma_collection()
    results = collection.query(query_texts=[query], n_results=top_k)
    chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    return list(zip(chunks, sources))


def generate_answer(query: str, retrieved_chunks):
    """Build a grounded prompt from retrieved chunks and call a local LLM via Ollama."""
    context = "\n\n".join([f"[Source: {src}]\n{chunk}" for chunk, src in retrieved_chunks])

    prompt = f"""You are a helpful college helpdesk assistant. Answer the student's
question using ONLY the context provided below. If the answer isn't in the
context, say you don't have that information and suggest they contact the
academic office directly. Keep answers concise and clear.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def ask(query: str, top_k: int = TOP_K):
    """End-to-end convenience function: retrieve + generate."""
    retrieved = retrieve(query, top_k=top_k)
    answer = generate_answer(query, retrieved)
    return answer, retrieved


if __name__ == "__main__":
    # Quick CLI test
    build_index()
    q = "What is the minimum attendance required for exams?"
    ans, sources = ask(q)
    print("\nQ:", q)
    print("A:", ans)
    print("\nRetrieved from:", [s for _, s in sources])
