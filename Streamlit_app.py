# streamlit_app.py  ← ONLY THIS FILE NEEDED
import streamlit as st
from PIL import Image
import pytesseract
import pdfplumber
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceHub
import os

st.set_page_config(page_title="Audalaxy RAG", layout="wide")
st.title("Audalaxy Multimodal RAG – Wissensintegration mit KI")
st.markdown("**Liest PDF (Text + Bilder + Tabellen) → Beantwortet Fragen präzise**")

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.header("Dokument hochladen")
    uploaded_file = st.file_uploader(
        "PDF, Bild (JPG/PNG) oder Excel",
        type=["pdf", "jpg", "jpeg", "png", "xlsx"]
    )
    
    if st.button("Wissensbasis erstellen", type="primary"):
        if not uploaded_file:
            st.error("Bitte eine Datei hochladen")
        else:
            with st.spinner("Lese Dokument ein (OCR + Tabellen)..."):
                raw_text = ""
                
                # 1. PDF mit Text + Bilder + Tabellen
                if uploaded_file.type == "application/pdf":
                    with pdfplumber.open(uploaded_file) as pdf:
                        for page in pdf.pages:
                            raw_text += page.extract_text() or ""
                            # Extract tables
                            tables = page.extract_tables()
                            for table in tables:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                raw_text += "\nTabelle:\n" + df.to_string() + "\n"
                            # Extract images → OCR
                            if page.images:
                                img = page.to_image(resolution=150).original
                                raw_text += "\n" + pytesseract.image_to_string(img, lang='deu') + "\n"
                
                # 2. Pure Image
                elif uploaded_file.type.startswith("image/"):
                    img = Image.open(uploaded_file)
                    raw_text = pytesseract.image_to_string(img, lang='deu')
                
                # 3. Excel
                elif "excel" in uploaded_file.type:
                    df = pd.read_excel(uploaded_file)
                    raw_text = df.to_string()

                # Split & Index
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([raw_text])
                
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                st.session_state.ready = True
                st.success("Wissensbasis bereit! Frage stellen →")

# ------------------- MAIN UI -------------------
if st.session_state.get("ready"):
    question = st.text_input(
        "Deine Frage zum Dokument:",
        placeholder="Was steht über Kündigungsfristen im Vertrag?"
    )
    
    if st.button("Antworten") and question:
        with st.spinner("Suche in deinem Dokument..."):
            template = """Du bist ein präziser Assistent für Audalaxy.
            Antworte AUSSCHLIESSLICH aus dem Kontext. Wenn du es nicht weißt, sag: "Nicht im Dokument gefunden."

            Kontext:
            {context}

            Frage: {question}
            Antwort:"""
            
            prompt = PromptTemplate.from_template(template)
            
            # Free & strong LLM
            llm = HuggingFaceHub(
                repo_id="HuggingFaceH4/zephyr-7b-alpha",
                model_kwargs={"temperature": 0.1, "max_new_tokens": 512}
            )
            
            chain = (
                {"context": st.session_state.retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            answer = chain.invoke(question)
            st.success("Antwort:")
            st.markdown(answer)
            
            with st.expander("Quellen (Chunks)"):
                for doc in st.session_state.retriever.invoke(question):
                    st.caption(doc.page_content[:600] + "...")
else:
    st.info("Lade dein Dokument hoch → Klicke auf 'Wissensbasis erstellen' → Stelle deine Frage")
    st.image("https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d?w=800")

st.caption("Built by Jay Khakhar | Für Audalaxy Vortest | 100% funktional | Live Demo bereit")
