import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv
import faiss
import numpy as np
from fastembed import TextEmbedding
import numpy as np

os.environ["PYTORCH_DISABLE_SDPA"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# ----------------------------------------
# Directories (modify if needed)
# ----------------------------------------
MD_DIR = "project\out_md"   # Folder where your Markdown files live

# ----------------------------------------
# 1. Setup
# ----------------------------------------
load_dotenv()
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY missing in .env")

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# Local embeddings model
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
embedder = TextEmbedding(model_name=EMBED_MODEL_NAME)

# ----------------------------------------
# 2. Load all Markdown files
# ----------------------------------------
def load_all_markdown(md_dir):
    full_text = ""

    if not os.path.exists(md_dir):
        raise FileNotFoundError(f"Markdown directory not found: {md_dir}")

    md_files = [f for f in os.listdir(md_dir) if f.lower().endswith(".md")]

    if not md_files:
        raise ValueError("No markdown (.md) files found in output_md/ folder.")

    for fname in md_files:
        path = os.path.join(md_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            full_text += f"\n\n# FILE: {fname}\n\n"
            full_text += f.read()

    return full_text


# ----------------------------------------
# 3. Chunk text
# ----------------------------------------
def chunk_text(text, chunk_size=1500, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


# ----------------------------------------
# 4. FAISS Index
# ----------------------------------------
def embed_texts(texts):
    """
    Use fastembed to embed a list of texts and return a NumPy array of shape (n, dim).
    """
    # fastembed returns a generator of vectors; we convert to a list, then np.array
    return np.array(list(embedder.embed(texts)))


def build_faiss_index(chunks):
    """Create embeddings for chunks and build a FAISS index."""
    embeddings = embed_texts(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings


def retrieve_relevant_chunks(question, chunks, embeddings, index, k=5):
    """Given a question, find top-k most similar chunks."""
    q_emb = embed_texts([question])  # shape (1, dim)
    distances, indices = index.search(q_emb, k)
    indices = indices[0]

    selected_chunks = [chunks[i] for i in indices if i < len(chunks)]
    return selected_chunks



# ----------------------------------------
# 6. OpenRouter API call
# ----------------------------------------
def call_llm(context, query):
    data = {
        "model": "mistralai/mistral-7b-instruct:free",
        "temperature": 0.0,
        "max_tokens": 1000,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful assistant answering questions ONLY from the provided context.\n"
                    "- If the context contains an explicit list, heading, or bullet points that answer the question, "
                    "you MUST reproduce that list as-is (same items, no extra categories).\n"
                    "- Do NOT invent new terms, categories, or characteristics.\n"
                    "- Use only concepts and wording that appear in the context, with very minor rephrasing if needed.\n"
                    "- If the answer is incomplete or missing in the context, say: "
                    "'I don't know based on the provided documents.'"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer concisely:",
            },
        ],
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=HEADERS,
        json=data,
    )

    if response.status_code != 200:
        return f"⚠️ API Error {response.status_code}: {response.text}"

    return response.json()["choices"][0]["message"]["content"].strip()


# ----------------------------------------
# 7. Streamlit App (no uploads)
# ----------------------------------------
def main():
    st.set_page_config(page_title="UNICRI Toolkit Chatbot", page_icon="🛡️")
    st.title("AI Toolkit Chatbot")

    # Initialize state
    if "index_built" not in st.session_state:
        st.session_state.index_built = False

    # Build index once
    if not st.session_state.index_built:
        st.write("🔄 Loading markdown files into memory...")
        full_text = load_all_markdown(MD_DIR)

        st.write("🔄 Chunking data...")
        chunks = chunk_text(full_text)

        st.write("🔄 Building FAISS vector index...")
        index, embeddings = build_faiss_index(chunks)

        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.index = index
        st.session_state.index_built = True

        st.success("🚀 Knowledge index built successfully!")

    st.subheader("💬 Ask a question")
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                context_chunks = retrieve_relevant_chunks(
                    question,
                    st.session_state.chunks,
                    st.session_state.embeddings,
                    st.session_state.index,
                )
                context = "\n\n---\n\n".join(context_chunks)
                answer = call_llm(context, question)

            st.markdown("### 🧠 Answer:")
            st.write(answer)

            with st.expander("🔍 Retrieved context"):
                st.write(context)


if __name__ == "__main__":
    main()
