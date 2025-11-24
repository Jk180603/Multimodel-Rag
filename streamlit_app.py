# streamlit_app.py → ONLY THIS FILE – INSTANT ANSWERS
import streamlit as st
from PIL import Image
import pytesseract
import pdfplumber
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import torch

st.set_page_config(page_title="MultiRAG Chat", layout="centered")
st.title("MultiRAG Chat – Instant Document Q&A")
st.caption("Upload PDF/Image → Ask instantly | Built for CV")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# FAST LOCAL LLM (answers in 1-2 seconds!)
@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_new_tokens=200,
        temperature=0.1
    )

llm = load_llm()

# Sidebar
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("PDF • Images • Screenshots", type=["pdf", "png", "jpg", "jpeg"])

    if st.button("Build Knowledge Base", type="primary"):
        if uploaded_file:
            with st.spinner("Reading document..."):
                text = ""

                if uploaded_file.type == "application/pdf":
                    with pdfplumber.open(uploaded_file) as pdf:
                        for page in pdf.pages:
                            text += (page.extract_text() or "") + "\n"
                            if page.images:
                                img = page.to_image(resolution=200).original
                                text += pytesseract.image_to_string(img) + "\n"

                elif uploaded_file.type.startswith("image/"):
                    img = Image.open(uploaded_file)
                    text = pytesseract.image_to_string(img)

                # Build vector DB
                splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
                chunks = splitter.create_documents([text])
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                st.success("Ready! Ask your question")
        else:
            st.error("Upload a file first")

# Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.retriever:
    if prompt := st.chat_input("Ask about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Answering..."):
                docs = st.session_state.retriever.invoke(prompt)
                context = "\n".join([d.page_content for d in docs])

                full_prompt = f"Context: {context}\n\nQuestion: {prompt}\nAnswer:"

                result = llm(full_prompt)
                answer = result[0]["generated_text"].split("Answer:")[-1].strip()

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload your document → Click 'Build Knowledge Base'")
    st.image("https://images.unsplash.com/photo-1504639725590-34d0984388bd?w=800")

st.caption("Built by Jay Khakhar | Instant Answers | Works with Screenshots | Top CV Project")
