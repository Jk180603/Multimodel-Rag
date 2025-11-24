import streamlit as st
from PIL import Image
import pytesseract
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

############################################
# FASTEST POSSIBLE SETTINGS
############################################

# Use smaller + faster LLM instead of FLAN-T5-LARGE
@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",   # <<< MUCH FASTER
        max_new_tokens=120,
        device=-1                      # auto CPU/GPU
    )

# Faster embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
llm = load_llm()

############################################
# STREAMLIT UI
############################################

st.set_page_config(page_title="MultiRAG Chat (Ultra-Fast)", layout="centered")
st.title("⚡ MultiRAG Chat – Instant Answers from Documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []


############################################
# FILE UPLOADING + PROCESSING
############################################

with st.sidebar:
    st.header("📄 Upload Document")
    uploaded = st.file_uploader("Upload PDF or Image", type=["pdf", "jpg", "jpeg", "png"])

    if st.button("🚀 Build Knowledge Base"):
        if not uploaded:
            st.error("Upload a file first")
        else:
            with st.spinner("Extracting and indexing..."):

                text = ""

                # PDF PROCESSING
                if uploaded.type == "application/pdf":
                    with pdfplumber.open(uploaded) as pdf:
                        for page in pdf.pages:
                            txt = page.extract_text()
                            if txt:
                                text += txt + "\n"

                # IMAGE PROCESSING
                else:
                    img = Image.open(uploaded)
                    text = pytesseract.image_to_string(img)

                # CLEANING
                text = " ".join(text.split())

                if len(text) < 20:
                    st.error("No readable text found!")
                else:
                    # CHUNKING (small chunks = faster embedding)
                    words = text.split()
                    chunk_size = 120      # smaller chunks = faster indexing
                    chunks = [
                        " ".join(words[i:i+chunk_size])
                        for i in range(0, len(words), chunk_size)
                    ]

                    embeddings = embedder.encode(chunks, batch_size=16)

                    dim = embeddings.shape[1]
                    index = faiss.IndexFlatL2(dim)
                    index.add(np.array(embeddings).astype("float32"))

                    st.session_state.index = index
                    st.session_state.chunks = chunks

                    st.success(f"Indexed {len(chunks)} chunks → Superfast ready!")


############################################
# CHAT INTERFACE
############################################

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.index is not None:
    prompt = st.chat_input("Ask something from your document...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                query_emb = embedder.encode([prompt])
                scores, ids = st.session_state.index.search(
                    np.array(query_emb).astype("float32"), k=3
                )

                context = "\n\n".join(
                    st.session_state.chunks[i] for i in ids[0]
                )

                final_prompt = (
                    f"Answer using ONLY this context:\n\n{context}\n\n"
                    f"Question: {prompt}\nAnswer:"
                )

                result = llm(final_prompt)[0]["generated_text"]
                answer = result.split("Answer:")[-1].strip()

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Upload a document and click **Build Knowledge Base**.")
