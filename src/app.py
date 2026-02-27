import streamlit as st
from rag_qa import build_qa_system

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
project_id = None

@st.cache_resource(show_spinner=False)
def load_system(files):
    return build_qa_system(files)

if uploaded_files:
    with st.spinner("Processing documents..."):
        qa, project_title, project_id = load_system(uploaded_files)
    st.success("Documents processed successfully! Ask your question below 👇")

query = st.text_input("❓ Ask a question")

if query and qa:
    q = query.lower()

    # 🎯 RULE-BASED ANSWERS (NO SOURCES)
    if "title" in q and project_title:
        st.subheader("✅ Answer")
        st.write(project_title)

    elif ("project id" in q or "projectid" in q) and project_id:
        st.subheader("✅ Answer")
        st.write(f"The project ID is {project_id}.")

    # 🤖 RAG-BASED ANSWERS (WITH SOURCES)
    else:
        try:
            with st.spinner("Thinking..."):
                result = qa(query)

            st.subheader("✅ Answer")
            st.write(result.get("result", "No answer found."))

            if result.get("source_documents"):
                st.subheader("📚 Sources")
                for i, _ in enumerate(result["source_documents"], 1):
                    st.write(f"Source {i}")

        except Exception:
            st.error(
                "The AI service is temporarily busy. "
                "Please wait a few seconds and try again."
            )