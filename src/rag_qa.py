import os
import tempfile
import re
from pypdf import PdfReader

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub


def extract_project_title(text):
    match = re.search(r'ON\s+["“](.+?)["”]', text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_project_id(text):
    match = re.search(r'Project\s*Id\s*:\s*(\d+)', text, re.IGNORECASE)
    return match.group(1) if match else None


def build_qa_system(uploaded_files):
    full_text = ""

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name

        reader = PdfReader(path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                full_text += t + "\n"

        os.remove(path)

    if not full_text.strip():
        raise ValueError("No readable text found in PDF.")

    project_title = extract_project_title(full_text)
    project_id = extract_project_id(full_text)

    documents = [Document(page_content=full_text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(docs, embedding=embeddings)

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",
        model_kwargs={"temperature": 0.1}
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Answer using ONLY the context below.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        )
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa, project_title, project_id