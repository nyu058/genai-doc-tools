from base.base import DocumentAiTool
import shutil
import os
from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class PdfAiTool(DocumentAiTool):

    prompt_template = """
You are a university student and is asked by the professor to write a summary of the uploaded document. 
The summary should have close to {word_approx} words but must not exceed {word_limit} words and {para_limit} paragraphs.  
Focus on capturing the main ideas and key points discussed in the document. Use your own words and layman's terms.
"""
    model = "gpt-4o-mini"

    def load_file(self, file_name, content):
        self.pages = self.process_file(file_name, content)

    @lru_cache
    def process_file(self, file_name, content):
        """process file content and save to a tmp location"""
        base_dir = os.path.abspath(os.path.dirname(__file__))
        try:
            os.mkdir(os.path.join(base_dir, "tmp"))
        except FileExistsError:
            pass
        upload_path = os.path.join(base_dir, "tmp", file_name)
        with open(upload_path, "wb") as f:
            f.write(content)
        loader = PyPDFLoader(upload_path)
        pages = loader.load_and_split()
        # cleaning up
        try:
            shutil.rmtree(os.path.join(base_dir, "tmp"))
        except FileNotFoundError:
            pass
        return pages
    
    def generate_response(self, **kwargs):
        question = self.prompt_template.format(**kwargs)
        faiss_index = FAISS.from_documents(self.pages, OpenAIEmbeddings())
        docs = faiss_index.similarity_search(question)

        llm = ChatOpenAI(temperature=0, model_name=self.model)
        chain = load_qa_chain(llm, chain_type="stuff")

        result = chain.run(input_documents=docs, question=question)
        return result
    
