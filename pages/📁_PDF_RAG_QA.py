import streamlit as st
from tools.rag_qa import RagAiTool
import os
import uuid

st.set_page_config(page_title="RAG QA", page_icon="📁")
REDIS_URL = os.environ.get("REDIS_URL")
rag = RagAiTool(REDIS_URL)

st.title("RAG QA")
# Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload File", type="pdf", accept_multiple_files=True)
if uploaded_file:
    with st.spinner("Processing file..."):
        for f in uploaded_file:
            rag.load_file(f.name, f.getvalue())
            st.success(f"File {f.name} processed!")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your question..."):
    st.session_state.messages.append({"role":"user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        stream = rag.qa(prompt, st.session_state.session_id)
        result = st.write_stream(stream)

    st.session_state.messages.append({"role":"assistant", "content": result})
