"""
app.py
Streamlit web app for AI Document Assistant.
"""

import streamlit as st
from rag_qa import load_qa_chain

# Load RetrievalQA chain
qa = load_qa_chain()

# Streamlit UI
st.title("📄 AI Document Assistant")
st.write("Upload PDFs into `data/` folder and ask questions below.")

# User input
query = st.text_input("❓ Ask a question from your documents:")

if query:
    result = qa({"query": query})
    st.write("### ✅ Answer:")
    st.write(result["result"])

    # Show sources
    st.write("### 📚 Sources:")
    for doc in result["source_documents"]:
        st.write(f"- {doc.metadata}")
