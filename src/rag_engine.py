"""
RAG Engine for querying external climate documents using Google Gemini.

This module handles:
1. Loading PDFs and Text files from `data/documents/`
2. Chunking text and creating vector embeddings (SentenceTransformers)
3. Storing/retrieving from a local ChromaDB instance
4. Sending retrieved context + user query to the Gemini API
"""

import os
import glob
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (API Key)
load_dotenv()

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "documents"
DB_DIR = BASE_DIR / "data" / ".chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Fast, free, local embedding model
LLM_MODEL = "gemini-2.5-flash" # Use supported flash model for this key


class RAGEngine:
    def __init__(self):
        """Initialises the embedding model, vector store, and LLM."""
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, 
            temperature=0.3, 
            google_api_key=os.getenv("GEMINI_API_KEY", "")
        )
        self.vector_store = self._init_vector_store()
        self.rag_chain = self._build_chain()

    def _load_documents(self):
        """Loads all PDFs and TXT files from the documents directory."""
        if not DOCS_DIR.exists():
            DOCS_DIR.mkdir(parents=True, exist_ok=True)
            return []

        documents = []
        
        # Load TXT files
        for txt_file in glob.glob(str(DOCS_DIR / "*.txt")):
            loader = TextLoader(txt_file, encoding='utf-8')
            documents.extend(loader.load())
            
        # Load PDF files
        for pdf_file in glob.glob(str(DOCS_DIR / "*.pdf")):
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
            
        return documents

    def _init_vector_store(self):
        """Creates or loads the Chroma database without re-embedding everything."""
        # Check if we already built the DB
        vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(DB_DIR)
        )
        
        # If DB already has documents in it, just return it directly! (Makes loading 10x faster)
        if vector_store._collection.count() > 0:
            return vector_store

        # Otherwise, this is the first time. We need to load and embed.
        documents = self._load_documents()
        if not documents:
            return vector_store # Empty store

        # 1. Chunk the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        # 2. Add to the vector store
        vector_store.add_documents(documents=chunks)
        return vector_store

    def _build_chain(self):
        """Constructs the LangChain retrieval and generation pipeline."""
        # 1. Define the system prompt
        system_prompt = (
            "You are an expert Climate Science Assistant for the Pune Climate Change Prediction Project. "
            "Use the following pieces of retrieved context to answer the user's question accurately. "
            "If you don't know the answer based on the context, just say that you don't know. "
            "Keep the answer concise, informative, and professional.\n\n"
            "Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 2. Create the chains
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3}) # Get top 3 chunks
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain

    def query(self, user_question: str) -> str:
        """Sends a query to the RAG pipeline and returns the string answer."""
        if not os.getenv("GEMINI_API_KEY"):
            return "Error: GEMINI_API_KEY is not set in the .env file."
            
        # Check if vector store is entirely empty
        if self.vector_store._collection.count() == 0:
            return "The Knowledge Base is empty. Please add some PDF or TXT files to the `data/documents/` folder."

        response = self.rag_chain.invoke({"input": user_question})
        return response["answer"]

# For quick testing during development
if __name__ == "__main__":
    engine = RAGEngine()
    result = engine.query("What major climate event happened in Pune in 2005?")
    print("-" * 40)
    print("Test Answer:", result)
    print("-" * 40)
