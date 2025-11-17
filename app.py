import os
import requests
import json
import streamlit as st
from os import getenv
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --------------------
# 1. Setup
# --------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# Embeddings model (CPU-friendly)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")


# --------------------
# 2. Read Markdown
# --------------------
def extract_text_from_markdown_files(uploaded_files):
    """Read all uploaded Markdown files and concatenate text"""
    full_text = ""
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        full_text += text + "\n"
    return full_text


# --------------------
# 3. Chunking
# --------------------
def chunk_text(text, chunk_size=1500, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


# --------------------
# 4. Build FAISS
# --------------------
def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings


# --------------------
# 5. Retrieval
# --------------------
def retrieve_relevant_chunks(question, chunks, embeddings, index, k=5):
    q_emb = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    indices = indices[0]
    return [chunks[i] for i in indices if i < len(chunks)]


# --------------------
# 6. LLM Call
# --------------------
def call_openrouter_llm(context, question):
    data = {
        "model": "mistralai/mistral-7b-instruct:free",
        "temperature": 0.2,
        "max_tokens": 500,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers ONLY using the provided context. "
                    "If the answer is not in the context, reply that you don't know."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            },
        ],
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=HEADERS,
        json=data,
    )

    if response.status_code != 200:
        return f"⚠️ OpenRouter error {response.status_code}: {response.text}"

    try:
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Unexpected response: {e}"


# --------------------
# 7. Streamlit UI
# --------------------
def main():
    st.set_page_config(page_title="Toolkit Chatbot", page_icon="🤖")
    st.title("🤖 Chat with the UNICRI + INTERPOL Toolkit (Markdown RAG)")

    st.write("Upload the Markdown files generated from your pipeline.")

    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None

    # Sidebar upload
    with st.sidebar:
        st.header("📄 Upload Toolkit Markdown Files")
        uploaded_files = st.file_uploader("Upload .md files", type="md", accept_multiple_files=True)

        if st.button("Process"):
            if not uploaded_files:
                st.warning("Please upload at least one Markdown file.")
            else:
                with st.spinner("Processing Markdown and building FAISS index…"):
                    text = extract_text_from_markdown_files(uploaded_files)
                    chunks = chunk_text(text)
                    index, embeddings = build_faiss_index(chunks)

                    st.session_state.faiss_index = index
                    st.session_state.chunks = chunks
                    st.session_state.embeddings = embeddings

                st.success("Markdown processed successfully! You can now ask questions.")

    # Question box
    st.subheader("💬 Ask a question")
    question = st.text_input("Ask something about the UNICRI Toolkit:")

    if st.button("Get Answer"):
        if st.session_state.faiss_index is None:
            st.error("Please upload and process Markdown files first.")
        else:
            with st.spinner("Thinking…"):
                relevant = retrieve_relevant_chunks(
                    question,
                    st.session_state.chunks,
                    st.session_state.embeddings,
                    st.session_state.faiss_index,
                )
                context = "\n\n---\n\n".join(relevant)
                answer = call_openrouter_llm(context, question)

            st.markdown("### 🧠 Answer")
            st.write(answer)

            with st.expander("🔎 Show retrieved context"):
                st.write(context)


if __name__ == "__main__":
    main()
