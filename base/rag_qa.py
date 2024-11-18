import os
import shutil
from base.base import AiTool
from functools import lru_cache
from langchain_community.document_loaders import PyPDFLoader
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RagAiTool(AiTool):

    def __init__(self, redis_url):
        config = RedisConfig(index_name="document", redis_url=redis_url)
        self.vector_store = RedisVectorStore(
            OpenAIEmbeddings(model="text-embedding-3-large"), config=config
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.vector_store.as_retriever(), contextualize_q_prompt
        )
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
        self.store = {}
        self.conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        super().__init__()

    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    @lru_cache
    def load_file(self, file_name, content):
        """load and process file and save to a vector DB"""
        base_dir = os.path.abspath(os.path.dirname(__file__))
        try:
            os.mkdir(os.path.join(base_dir, "tmp"))
        except FileExistsError:
            pass
        upload_path = os.path.join(base_dir, "tmp", file_name)
        with open(upload_path, "wb") as f:
            f.write(content)
        loader = PyPDFLoader(upload_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
        pages = loader.load_and_split(text_splitter=splitter)
        # cleaning up
        try:
            shutil.rmtree(os.path.join(base_dir, "tmp"))
        except FileNotFoundError:
            pass
        self.vector_store.add_documents(pages)

    def qa(self, query, session_id):
        config = {"configurable":{"session_id":session_id}}
        response = self.conversational_rag_chain.stream({"input": query}, config=config)
        for res in response:
            if "answer" in res:
                yield res["answer"]
