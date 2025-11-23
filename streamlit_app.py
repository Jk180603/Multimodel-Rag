# streamlit_app.py → Paste this exactly
import streamlit as st
from PIL import Image
import pytesseract
import pdfplumber
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceHub
import os

st.set_page_config(page_title="MultiRAG Chat", layout="centered")
st.title("MultiRAG Chat – Ask Anything From Your Documents")
st.caption("Upload PDF, Images, Excel → Chat with your files | Built for CV & Portfolio")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Sidebar
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "PDF • Images • Excel • Screenshots",
        type=["pdf", "png", "jpg", "jpeg", "xlsx", "csv"]
    )
    
    if st.button("Build Knowledge Base", type="primary"):
        if uploaded_file:
            with st.spinner("Reading document with OCR + Tables..."):
                text = ""
                
                # Handle PDF
                if uploaded_file.type == "application/pdf":
                    with pdfplumber.open(uploaded_file) as pdf:
                        for page in pdf.pages:
                            text += (page.extract_text() or "") + "\n"
                            for table in page.extract_tables():
                                df = pd.DataFrame(table[1:], columns=table[0])
                                text += df.to_string() + "\n"
                            if page.images:
                                img = page.to_image(resolution=200).original
                                text += pytesseract.image_to_string(img) + "\n"
                
                # Handle Images (like your screenshot)
                elif uploaded_file.type.startswith("image/"):
                    img = Image.open(uploaded_file)
                    text = pytesseract.image_to_string(img)
                
                # Handle Excel
                else:
                    df = pd.read_excel(uploaded_file)
                    text = df.to_string()

                # Build vector DB
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.create_documents([text])
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                st.success("Ready! Now chat with your document")
        else:
            st.error("Please upload a file")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if st.session_state.retriever:
    if prompt := st.chat_input("Ask anything about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                template = """Answer only from this context. If not found, say "Not in document."
                Context: {context}
                Question: {question}
                Answer:"""
                prompt_template = PromptTemplate.from_template(template)
                
                llm = HuggingFaceHub(
                    repo_id="HuggingFaceH4/zephyr-7b-alpha",
                    model_kwargs={"temperature": 0.1}
                )
                
                chain = (
                    {"context": st.session_state.retriever, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )
                
                response = chain.invoke(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Upload a document first → Click 'Build Knowledge Base' → Start chatting")

st.markdown("---")
st.caption("Built by Jay Khakhar | Top 1% CV Project | Handles Images, Screenshots, Scanned PDFs")
