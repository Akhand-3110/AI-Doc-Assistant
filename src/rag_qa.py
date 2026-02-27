import os
import re
import tempfile
import time
import requests
from typing import Optional, List
from pypdf import PdfReader

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.language_models.llms import LLM


# -------------------------
# Rule-based extractors
# -------------------------
def extract_project_title(text):
    match = re.search(r'ON\s+["“](.+?)["”]', text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_project_id(text):
    match = re.search(r'Project\s*Id\s*:\s*(\d+)', text, re.IGNORECASE)
    return match.group(1) if match else None


# -------------------------
# Hugging Face Inference LLM
# -------------------------
class HFInferenceLLM(LLM):
    model: str = "google/flan-t5-small"

    @property
    def _llm_type(self) -> str:
        return "hf_inference"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            return "Hugging Face API token is not configured."

        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {token}"}

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 128,
                "temperature": 0.1
            }
        }

        # Retry once (HF often returns 503)
        for attempt in range(2):
            try:
                r = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )

                if r.status_code == 503:
                    time.sleep(5)
                    continue

                r.raise_for_status()
                return r.json()[0]["generated_text"]

            except requests.exceptions.RequestException:
                if attempt == 1:
                    return (
                        "The AI model is temporarily busy. "
                        "Please wait a few seconds and try again."
                    )


# -------------------------
# Main QA builder
# -------------------------
def build_qa_system(uploaded_files):
    full_text = ""

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        reader = PdfReader(path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        os.remove(path)

    if not full_text.strip():
        raise ValueError("No readable text found in uploaded PDF.")

    # Rule-based metadata
    project_title = extract_project_title(full_text)
    project_id = extract_project_id(full_text)

    documents = [Document(page_content=full_text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(chunks, embedding=embeddings)

    llm = HFInferenceLLM()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Answer the question using ONLY the context below.\n\n"
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