import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_FOLDER = "data"
DB_FOLDER = "db"

def ingest_data():
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma.from_documents(split_docs, embeddings, persist_directory=DB_FOLDER)
    db.persist()
    print(f"🎉 Ingestion complete! {len(split_docs)} chunks saved in Chroma DB.")

if __name__ == "__main__":
    ingest_data()