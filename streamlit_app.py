# streamlit_app.py → ONLY THIS FILE – 100% WORKING
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
from transformers import pipeline
import torch

st.set_page_config(page_title="MultiRAG Chat", layout="centered", page_icon="robot")
st.title("MultiRAG Chat – Talk to Your Documents")
st.caption("Upload PDF, Images, Excel, Screenshots → Chat with them | Perfect for CV")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Load local LLM (NO API KEY NEEDED)
@st.cache_resource
def load_llm():
    with st.spinner("Loading AI model (first time ~60 sec)..."):
        return pipeline(
            "text-generation",
            model="google/gemma-2b-it",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True
        )

llm = load_llm()

# Sidebar – Upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "PDF • Images • Excel • Screenshots",
        type=["pdf", "png", "jpg", "jpeg", "xlsx", "csv"]
    )
    
    if st.button("Build Knowledge Base", type="primary"):
        if uploaded_file:
            with st.spinner("Extracting text with OCR..."):
                text = ""
                
                if uploaded_file.type == "application/pdf":
                    with pdfplumber.open(uploaded_file) as pdf:
                        for page in pdf.pages:
                            text += (page.extract_text() or "") + "\n"
                            for table in page.extract_tables():
                                df = pd.DataFrame(table[1:], columns=table[0] if table else None)
                                text += df.to_string() + "\n"
                            if page.images:
                                img = page.to_image(resolution=200).original
                                text += pytesseract.image_to_string(img, lang='deu+eng') + "\n"
                
                elif uploaded_file.type.startswith("image/"):
                    img = Image.open(uploaded_file)
                    text = pytesseract.image_to_string(img, lang='deu+eng')
                
                else:
                    df = pd.read_excel(uploaded_file)
                    text = df.to_string()

                # Build vector DB
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = splitter.create_documents([text])
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                st.success("Ready! Start chatting")
        else:
            st.error("Please upload a file")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.retriever:
    if prompt := st.chat_input("Ask anything about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve context
                docs = st.session_state.retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Prompt
                full_prompt = f"""You are a helpful assistant. Answer ONLY from this context:
                
                {context}
                
                Question: {prompt}
                Answer:"""
                
                # Generate
                outputs = llm(full_prompt)
                response = outputs[0]["generated_text"].split("Answer:")[-1].strip()
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Upload your document → Click 'Build Knowledge Base' → Start chatting")
    st.image("https://images.unsplash.com/photo-1504639725590-34d0984388bd?w=800")

st.markdown("---")
st.caption("Built by Jay Khakhar | Works with Images & Screenshots | No API Key Needed | CV-Ready")
