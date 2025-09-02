import streamlit as st
import rag_qa

qa = rag_qa.load_qa_chain()

st.title("📄 AI Document Assistant")
st.write("Upload PDFs into `data/` folder and ask questions below.")

query = st.text_input("❓ Ask a question from your documents:")

if query:
    result = qa({"query": query})
    st.write("### ✅ Answer:")
    st.write(result["result"])

    st.write("### 📚 Sources:")
    for doc in result["source_documents"]:
        st.write(f"- {doc.metadata}")
