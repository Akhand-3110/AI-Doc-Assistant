from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

DB_FOLDER = "db"

def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    generator = pipeline("text2text-generation", model="google/flan-t5-base")

    def qa(query: str):
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])
        if not context.strip():
            return {"result": "⚠️ No relevant context found.", "source_documents": []}

        prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
        result = generator(prompt, max_length=256, do_sample=False)
        answer = result[0]['generated_text'].strip() if result else "❓ No answer generated."

        return {"result": answer, "source_documents": docs}

    return qa


def rag_qa():
    qa = load_qa_chain()

    print("\n🤖 Ask me anything about your documents! (type 'exit' to quit)\n")
    while True:
        query = input("👉 Your question: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        if not query.strip():
            print("⚠️ Please enter a valid question.")
            continue

        result = qa(query)
        print("\n📖 Answer:", result["result"], "\n")


if __name__ == "__main__":
    rag_qa()
