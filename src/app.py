import streamlit as st
from rag_qa import build_qa_chain_from_pdfs

st.set_page_config(page_title="AI Document Assistant", layout="centered")

st.title("📄 AI Document Assistant")
st.write("Upload one or more PDF files and ask questions from them.")

# 📤 File uploader
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

qa = None

if uploaded_files:
    with st.spinner("Processing documents..."):
        qa = build_qa_chain_from_pdfs(uploaded_files)
    st.success("Documents processed! Ask your question below 👇")

# 💬 Question input
query = st.text_input("❓ Ask a question")

if query:
    if qa is None:
        st.warning("Please upload PDF files first.")
    else:
        with st.spinner("Thinking..."):
            result = qa(query)

        st.subheader("✅ Answer")
        st.write(result["result"])

        if result.get("source_documents"):
            st.subheader("📚 Sources")
            for i, doc in enumerate(result["source_documents"], 1):
                st.write(f"Source {i}")