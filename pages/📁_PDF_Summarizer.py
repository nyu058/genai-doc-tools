import streamlit as st
from tools.pdf import PdfAiTool

st.set_page_config(page_title="PDF Summarizer", page_icon="📁")

pdf_ai = PdfAiTool()

st.title("PDF Summarizer")
# Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload File", type="pdf")
if uploaded_file:
    with st.spinner("Processing file..."):
        pdf_ai.load_file(uploaded_file.name, uploaded_file.getvalue())
    st.success("File processed!")

word_limit = st.sidebar.number_input("Word limit", value=300)
word_approx = st.sidebar.number_input("Approx. word count", value=200)
para_limit = st.sidebar.number_input("Paragraphs", value=1)
summarize = st.sidebar.button("Summarize", disabled=not bool(uploaded_file))
if summarize:
    with st.spinner("Generating summary..."):
        result = pdf_ai.generate_response(word_limit=word_limit, word_approx=word_approx, para_limit=para_limit)
    st.write(result)
