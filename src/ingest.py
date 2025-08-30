"""
Free Ingest Script (No OpenAI Needed)
This script:
1. Reads all PDFs from the 'data/' folder
2. Splits them into chunks
3. Creates embeddings using HuggingFace (local, free)
4. Saves them into a Chroma database (for later QA)
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Folder paths
DATA_FOLDER = "data"
DB_FOLDER = "db"

def ingest_data():
    """Loads PDFs, splits into chunks, creates embeddings, and saves them."""
    # 1. Collect PDF files
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️ No PDF files found in 'data/' folder. Please add some documents.")
        return

    all_docs = []
    for pdf in pdf_files:
        file_path = os.path.join(DATA_FOLDER, pdf)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"✅ Loaded {pdf}")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    # 3. Create free local embeddings (HuggingFace)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Save to Chroma DB
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=DB_FOLDER)
    db.persist()
    print(f"🎉 Ingestion complete! {len(split_docs)} chunks saved in Chroma DB.")

if __name__ == "__main__":
    ingest_data()
