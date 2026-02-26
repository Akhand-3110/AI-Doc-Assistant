from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import pipeline
from pypdf import PdfReader
import tempfile
import os


def build_qa_chain_from_pdfs(uploaded_files):
    documents = []

    # 🔹 Extract text safely
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        reader = PdfReader(tmp_path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        os.remove(tmp_path)

        # 🚫 Skip empty PDFs
        if text.strip():
            documents.append(Document(page_content=text))

    # 🚫 If no text extracted → STOP safely
    if not documents:
        raise ValueError(
            "No readable text found in uploaded PDF(s). "
            "Please upload text-based PDFs (not scanned images)."
        )

    # 🔹 Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # 🚫 Safety check
    if not docs:
        raise ValueError("Document splitting failed. PDF content may be empty.")

    # 🔹 Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(docs, embedding=embeddings)

    # 🔹 LLM
    hf_pipeline = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return qa