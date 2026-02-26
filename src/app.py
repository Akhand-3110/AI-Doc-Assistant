import streamlit as st
from rag_qa import load_qa_chain

qa = load_qa_chain()

st.title("📄 AI Document Assistant")
st.write("Upload PDFs into `data/` folder and ask questions below.")

query = st.text_input("❓ Ask a question from your documents:")

if query:
    result = qa(query)

    st.write("### ✅ Answer:")
    st.write(result["result"])

    if result.get("source_documents"):
        st.write("### 📚 Sources:")
        for doc in result["source_documents"]:
            st.write(f"- {doc.metadata}")