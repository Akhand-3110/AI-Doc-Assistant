import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


DB_DIR = "db"
DATA_DIR = "data"


def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 🔹 If DB does not exist, CREATE IT
    if not os.path.exists(DB_DIR) or len(os.listdir(DB_DIR)) == 0:
        loader = PyPDFLoader(os.path.join(DATA_DIR, "Akhand_resume.pdf.pdf"))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = splitter.split_documents(documents)

        vectordb = Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        vectordb.persist()
    else:
        vectordb = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )

    retriever = vectordb.as_retriever()

    hf_pipeline = pipeline(
        task="text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa