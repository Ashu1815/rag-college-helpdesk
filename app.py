"""
Streamlit chat interface for the College Helpdesk RAG chatbot.
Run with: streamlit run app.py
"""

import streamlit as st
from rag_pipeline import ask, build_index, DB_DIR
import os

st.set_page_config(page_title="College Helpdesk Bot", page_icon="🎓")
st.title("🎓 College Helpdesk Chatbot")
st.caption("RAG-powered assistant — answers grounded in official college documents, running fully locally.")

# Build the index automatically on first run if it doesn't exist yet
if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
    with st.spinner("Indexing documents for the first time..."):
        build_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about attendance, exams, hostel, certificates..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, retrieved = ask(prompt)
                st.markdown(answer)
                with st.expander("📄 Sources used"):
                    for chunk, source in retrieved:
                        st.markdown(f"**{source}**")
                        st.caption(chunk[:300] + ("..." if len(chunk) > 300 else ""))
            except Exception as e:
                answer = (
                    "⚠️ Couldn't reach the local model. Make sure Ollama is running "
                    f"and the model is pulled (`ollama pull llama3.2`).\n\nError: {e}"
                )
                st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
        **Architecture:**
        1. Documents chunked & embedded locally (`sentence-transformers`)
        2. Vectors stored in **ChromaDB** (local, persistent)
        3. Query embedded → top-matching chunks retrieved
        4. Chunks + query sent to a **local LLM via Ollama**
        5. Grounded answer returned with sources

        No API keys. No external calls. Fully offline after setup.
        """
    )
    if st.button("🔄 Rebuild index"):
        with st.spinner("Rebuilding..."):
            build_index()
        st.success("Index rebuilt!")
