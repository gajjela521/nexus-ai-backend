import os
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Enterprise Grade Configuration
# Using a robust local model for embeddings (Free, no API key needed for this part)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DB_PATH = "faiss_index"

class RAGService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        self.vector_store = self._load_or_create_vector_store()
        # Initialize LLM (Will fallback gracefully if no key)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=api_key)
        else:
            self.llm = None  # Will handle "Mock" generation if no key

    def _load_or_create_vector_store(self):
        if os.path.exists(VECTOR_DB_PATH):
            return FAISS.load_local(VECTOR_DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
        # Initialize empty if not exists
        return None

    def ingest_documents(self, documents: List[Dict[str, str]]):
        """
        Ingest text data into Vector DB.
        Standard Flow: Text -> Split -> Embed -> Index
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        docs = []
        for d in documents:
            chunks = text_splitter.split_text(d["content"])
            for chunk in chunks:
                docs.append(Document(page_content=chunk, metadata=d.get("metadata", {})))
        
        if self.vector_store:
            self.vector_store.add_documents(docs)
        else:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            
        # Persist to disk (Enterprise requirement)
        self.vector_store.save_local(VECTOR_DB_PATH)
        return len(docs)

    def query(self, query_text: str, api_key: str = None) -> Dict:
        """
        RAG Flow: Query -> Vector Search -> Context Integration -> LLM Answer
        Allows dynamic API Key for session-based usage.
        """
        if not self.vector_store:
            return {"answer": "Knowledge base is empty. Please ingest data first.", "sources": []}

        # 1. Retrieval (Get accurate points)
        # Fetch top 3 relevant chunks
        docs = self.vector_store.similarity_search(query_text, k=3)
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # 2. Determine LLM to use (Session Key > Env Key > None)
        active_llm = self.llm
        if api_key:
            # Create temporary LLM instance for this request
            active_llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=api_key)
        
        # 3. Generation (LLM)
        if active_llm:
            prompt = PromptTemplate(
                template="""Use the following pieces of context to answer the question at the end. 
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                
                Context:
                {context}
                
                Question: {question}
                Answer:""",
                input_variables=["context", "question"]
            )
            chain = RetrievalQA.from_chain_type(
                llm=active_llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(),
                chain_type_kwargs={"prompt": prompt}
            )
            answer = chain.run(query_text)
        else:
            # Fallback for Demo without API Key
            answer = f"**[Simulated LLM Response]**\nBased on the analysis, here are the relevant points found in the database:\n\n{context_text}\n\n*(To get real GPT-4 generation, provide an API Key in the request or set env var)*"

        return {
            "query": query_text,
            "answer": answer,
            "context_used": [d.page_content for d in docs]
        }

# Singleton Instance
rag_service = RAGService()
