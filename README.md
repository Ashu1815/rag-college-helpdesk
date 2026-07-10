# 🎓 College Helpdesk RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers student questions
using official college documents (attendance rules, exam policy, hostel
allocation, certificates, etc.) — running **entirely locally, with no API
keys or paid services**.

## How it works

```
User question
     │
     ▼
Embed question (sentence-transformers, local)
     │
     ▼
Search ChromaDB vector store  ──────► Top-K relevant document chunks
     │
     ▼
Build prompt = question + retrieved chunks
     │
     ▼
Local LLM (Ollama, e.g. llama3.2) generates a grounded answer
     │
     ▼
Answer + cited sources shown in chat UI
```

**Why RAG instead of just asking an LLM directly?** The base model has never
seen your college's specific attendance policy or exam rules. RAG retrieves
the exact relevant text from your documents and feeds it to the model as
context, so answers are accurate and grounded instead of hallucinated.

## Tech stack

| Component        | Tool                                   |
|-------------------|----------------------------------------|
| Embeddings        | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector database   | ChromaDB (local, persistent)           |
| LLM               | Ollama (runs any local model, e.g. llama3.2, phi3, mistral) |
| UI                | Streamlit                              |

Everything runs on your own machine — no OpenAI/Anthropic API key needed,
which makes this free to run and easy to demo offline (e.g. during a viva or
interview with no internet).

## Setup

### 1. Install Ollama (the local LLM runner)
Download from https://ollama.com and install it. Then pull a model:
```bash
ollama pull llama3.2
```
(Any model works — `phi3`, `mistral`, `gemma2` etc. Just update `LLM_MODEL`
in `rag_pipeline.py` to match.)

### 2. Set up the Python environment
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your documents
Drop `.txt` files into `data/sample_docs/`. Two sample files (college FAQ
and exam rules) are already included so you can test immediately. To use
your real college's PDFs, convert them to `.txt` first (or extend
`rag_pipeline.py`'s `load_documents()` to parse PDFs with `pypdf`).

### 4. Build the vector index
```bash
python ingest.py
```

### 5. Run the chatbot
```bash
streamlit run app.py
```
Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Project structure
```
rag-college-helpdesk/
├── data/sample_docs/     # Source documents (FAQ, exam rules, etc.)
├── vectorstore/          # Auto-generated ChromaDB storage (created on first run)
├── rag_pipeline.py       # Core RAG logic: chunk, embed, retrieve, generate
├── ingest.py             # Script to (re)build the vector index
├── app.py                # Streamlit chat UI
├── requirements.txt
└── README.md
```

## Ideas to extend this for your resume project

- **PDF support**: parse real college PDFs (syllabus, handbook) using `pypdf`
- **Better chunking**: switch to sentence/paragraph-aware chunking instead of fixed character windows
- **Evaluation**: add a small test set of Q&A pairs and measure retrieval accuracy
- **Swap the LLM backend**: add an option to use OpenAI/Claude API for comparison, and discuss the local-vs-API tradeoffs in your report
- **Deploy it**: host the Streamlit app (Streamlit Community Cloud, or a small VM) and link it live on your resume
- **Add authentication** so it's a proper "college portal" style app

## What to say about this in interviews

Be ready to explain: why chunking size/overlap matters, what embeddings
actually represent, why retrieval beats fine-tuning for this use case, and
what happens when the retrieved context doesn't contain the answer (the
prompt explicitly instructs the model to say so rather than hallucinate).
