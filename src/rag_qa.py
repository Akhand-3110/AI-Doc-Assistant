import os
import tempfile
import re
from pypdf import PdfReader
from transformers import pipeline

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline


# 🔹 Rule-based extractors
def extract_project_title(text):
    match = re.search(r'ON\s+["“](.+?)["”]', text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_project_id(text):
    match = re.search(r'Project\s*Id\s*:\s*(\d+)', text, re.IGNORECASE)
    return match.group(1) if match else None


def build_qa_system(uploaded_files):
    full_text = ""

    # 🔹 Read PDFs safely
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        reader = PdfReader(tmp_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

        os.remove(tmp_path)

    if not full_text.strip():
        raise ValueError("No readable text found in uploaded PDF.")

    # 🔹 Rule-based fields
    project_title = extract_project_title(full_text)
    project_id = extract_project_id(full_text)

    # 🔹 Prepare documents
    documents = [Document(page_content=full_text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,     # memory-safe
        chunk_overlap=40
    )
    docs = splitter.split_documents(documents)

    # 🔹 Embeddings (lightweight)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(docs, embedding=embeddings)

    # 🔹 Lightweight LLM (CRITICAL)
    hf_pipeline = pipeline(
        task="text2text-generation",
        model="google/flan-t5-small",   # memory safe
        max_new_tokens=128
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # 🔹 Strong prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Answer the question using ONLY the context below.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer clearly and concisely."
        )
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain, project_title, project_id