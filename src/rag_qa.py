"""
Free RAG QA Script (No OpenAI Needed)
This script:
1. Loads the Chroma database created during ingest
2. Retrieves the most relevant chunks for a user query
3. Uses a HuggingFace model (local) to generate an answer
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

DB_FOLDER = "db"

def rag_qa():
    # 1. Load embeddings (must match ingest.py)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Load Chroma DB
    db = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)

    # 3. Create retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 4. Local HuggingFace text generation model
    generator = pipeline("text2text-generation", model="facebook/bart-large-cnn")

    print("\n🤖 Ask me anything about your documents! (type 'exit' to quit)\n")
    while True:
        query = input("👉 Your question: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # 5. Retrieve relevant docs
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 6. Generate answer with context
        prompt = f"Answer the question based only on the context:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        result = generator(prompt, max_length=300, do_sample=False)

        print("\n📖 Answer:", result[0]['generated_text'], "\n")

if __name__ == "__main__":
    rag_qa()
