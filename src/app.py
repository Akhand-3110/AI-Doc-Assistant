import streamlit as st
from rag_qa import build_qa_chain_from_pdfs

st.set_page_config(page_title="AI Document Assistant")

st.title("📄 AI Document Assistant")
st.write("Upload text-based PDF files and ask questions from them.")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

qa = None
project_title = None

if uploaded_files:
    with st.spinner("Processing documents..."):
        qa, project_title = build_qa_chain_from_pdfs(uploaded_files)
    st.success("Documents processed successfully! Ask your question below 👇")

query = st.text_input("❓ Ask a question")

if query and qa:
    # 🎯 SPECIAL HANDLING FOR TITLE QUESTIONS
    if "title" in query.lower() and project_title:
        st.subheader("✅ Answer")
        st.write(project_title)
    else:
        with st.spinner("Thinking..."):
            result = qa(query)

        st.subheader("✅ Answer")
        st.write(result.get("result", "No answer found."))

        if result.get("source_documents"):
            st.subheader("📚 Sources")
            for i, _ in enumerate(result["source_documents"], 1):
                st.write(f"Source {i}")