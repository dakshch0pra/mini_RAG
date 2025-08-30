# from langchain.chains import RetrievalQA
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores import Pinecone
# from langchain_community.document_loaders import TextLoader
# import os, re
# from langchain_core.documents import Document
# from langchain_cohere import CohereRerank
# from langchain.schema import HumanMessage, SystemMessage
# Set your Google API key (replace with yours)
# os.environ["GOOGLE_API_KEY"] = "AIzaSyCITsocNmlmc2WE2EzvffSgrGElX153Nak"
# os.environ["COHERE_API_KEY"] = "AejGUl4A1TosuUZFoeyiNixW65zEvNNUbeB3dFkf"

#Working Class Based Rag System

# import os
# from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_cohere import CohereRerank
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.schema import Document
# from typing import Dict, Any, List

# class RAGPipeline:
#     def __init__(self):
#         self.embeddings, self.llm, self.text_splitter, self.reranker = self._setup_components()
#         self.vector_store = None
    
#     def _setup_components(self):
#         # Set up API keys (replace with your actual keys)
#         os.environ["GOOGLE_API_KEY"] = "AIzaSyCITsocNmlmc2WE2EzvffSgrGElX153Nak"
#         os.environ["COHERE_API_KEY"] = "AejGUl4A1TosuUZFoeyiNixW65zEvNNUbeB3dFkf"
        
#         # Initialize components
#         embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/text-embedding-004"
#         )
        
#         llm = GoogleGenerativeAI(
#             model="gemini-2.0-flash",
#             temperature=0.2
#         )
        
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1200,
#             chunk_overlap=120,
#             separators=["\n\n", "\n", " ", ""]
#         )
        
#         reranker = CohereRerank(
#             model="rerank-english-v3.0",
#             top_n=5
#         )
        
#         return embeddings, llm, text_splitter, reranker
    
#     def create_knowledge_base(self, text_input: str):
#         # Split text into documents
#         docs = self.text_splitter.create_documents([text_input])
        
#         # Create FAISS vector store
#         self.vector_store = FAISS.from_documents(
#             documents=docs,
#             embedding=self.embeddings
#         )
        
#         print(f"Knowledge base created with {len(docs)} documents")
#         return self.vector_store
    
#     def add_text(self, new_text: str):
#         if self.vector_store is None:
#             # If no knowledge base exists, create one
#             return self.create_knowledge_base(new_text)
        
#         # Split new text into documents
#         new_docs = self.text_splitter.create_documents([new_text])
        
#         # Add to existing vector store
#         self.vector_store.add_documents(new_docs)
        
#         print(f"Added {len(new_docs)} new documents to knowledge base")
#         return self.vector_store
    
#     def query(self, question: str) -> Dict[str, Any]:
#         if self.vector_store is None:
#             return {
#                 "query": question,
#                 "answer": "No knowledge base found. Please create one first.",
#                 "sources": [],
#                 "num_sources": 0
#             }
        
#         # Step 1: Similarity search
#         similar_docs = self.vector_store.similarity_search(
#             question, 
#             k=10  # Get top 10 similar documents
#         )
        
#         if not similar_docs:
#             return {
#                 "query": question,
#                 "answer": "No relevant information found.",
#                 "sources": [],
#                 "num_sources": 0
#             }
        
#         # Step 2: Rerank documents using Cohere
#         reranked_docs = self.reranker.compress_documents(
#             documents=similar_docs,
#             query=question
#         )
        
#         # Step 3: Prepare context for LLM
#         context = "\n\n".join([doc.page_content for doc in reranked_docs])
        
#         # Step 4: Generate response
#         prompt = f"""Use the following context to answer the question. If you cannot answer based on the context, say so clearly.

# Context:
# {context}

# Question: {question}

# Answer:"""
        
#         response = self.llm.invoke(prompt)
        
#         # Prepare sources information
#         sources = []
#         for i, doc in enumerate(reranked_docs):
#             sources.append({
#                 "index": i + 1,
#                 "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
#                 "metadata": getattr(doc, 'metadata', {})
#             })
        
#         return {
#             "query": question,
#             "answer": response,
#             "sources": sources,
#             "num_sources": len(sources)
#         }

# # Example usage:
# if __name__ == "__main__":
#     # Initialize RAG pipeline
#     rag = RAGPipeline()
    
#     # Create knowledge base
#     print("=== RAG System Setup ===")
#     initial_text = input("Enter your initial knowledge text: ")
#     rag.create_knowledge_base(initial_text)
    
#     # Interactive query loop
#     print("\n=== Ask Questions (type 'quit' to exit) ===")
    
#     while True:  # This creates an infinite loop
#         question = input("\nYour question: ")
        
#         # Check if user wants to exit
#         if question.lower() in ['quit', 'exit', 'q']:
#             print("Goodbye!")
#             break  # This exits the loop
        
#         # Process the question
#         result = rag.query(question)
#         print(f"\nAnswer: {result['answer']}")
#         print(f"Sources used: {result['num_sources']}")


# FastAPI version (optional)

# main.py
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Your existing imports
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_cohere import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

import fitz

# ============= LOGGING SETUP =============
# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_app.log'),  # Saves logs to file
        logging.StreamHandler()  # Prints logs to console
    ]
)
logger = logging.getLogger(__name__)

# ============= PYDANTIC MODELS (Request/Response Schemas) =============
class TextInput(BaseModel):
    """Schema for text input requests"""
    text: str = Field(..., min_length=1, description="Text content to add to knowledge base")

class QueryInput(BaseModel):
    """Schema for query requests"""
    question: str = Field(..., min_length=1, description="Question to ask the RAG system")

class QueryResponse(BaseModel):
    """Schema for query responses"""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    num_sources: int
    timestamp: str

class StatusResponse(BaseModel):
    """Schema for status responses"""
    message: str
    status: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None

# ============= YOUR RAG CLASS (Enhanced with Logging) =============
class RAGPipeline:
    def __init__(self):
        logger.info("Initializing RAG Pipeline...")
        try:
            self.embeddings, self.llm, self.text_splitter, self.reranker = self._setup_components()
            self.vector_store = None
            logger.info("RAG Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {str(e)}")
            raise
    
    def _setup_components(self):
        logger.info("🔧 Setting up RAG components...")
        
        # Set up API keys
        os.environ["GOOGLE_API_KEY"] = "AIzaSyCITsocNmlmc2WE2EzvffSgrGElX153Nak"
        os.environ["COHERE_API_KEY"] = "AejGUl4A1TosuUZFoeyiNixW65zEvNNUbeB3dFkf"
        
        try:
            # Initialize components
            logger.info("Initializing embeddings...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004"
            )
            
            logger.info("Initializing LLM...")
            llm = GoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.2
            )
            
            logger.info("Initializing text splitter...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=120,
                separators=["\n\n", "\n", " ", ""]
            )
            
            logger.info("Initializing reranker...")
            reranker = CohereRerank(
                model="rerank-english-v3.0",
                top_n=5
            )
            
            logger.info("All components initialized successfully")
            return embeddings, llm, text_splitter, reranker
            
        except Exception as e:
            logger.error(f"Component initialization failed: {str(e)}")
            raise
    
    def create_knowledge_base(self, text_input: str):
        logger.info(f"Creating knowledge base with text length: {len(text_input)}")
        try:
            # Split text into documents
            docs = self.text_splitter.create_documents([text_input])
            logger.info(f"Text split into {len(docs)} documents")
            
            # Create FAISS vector store
            logger.info("Creating FAISS vector store...")
            self.vector_store = FAISS.from_documents(
                documents=docs,
                embedding=self.embeddings
            )
            
            logger.info(f"Knowledge base created with {len(docs)} documents")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to create knowledge base: {str(e)}")
            raise
    
    def add_text(self, new_text: str):
        logger.info(f"Adding new text (length: {len(new_text)})")
        try:
            if self.vector_store is None:
                logger.info("No existing knowledge base, creating new one...")
                return self.create_knowledge_base(new_text)
            
            # Split new text into documents
            new_docs = self.text_splitter.create_documents([new_text])
            logger.info(f"New text split into {len(new_docs)} documents")
            
            # Add to existing vector store
            self.vector_store.add_documents(new_docs)
            
            logger.info(f"Added {len(new_docs)} new documents to knowledge base")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to add text: {str(e)}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        logger.info(f"Processing query: {question[:100]}...")
        
        if self.vector_store is None:
            logger.warning("No knowledge base found")
            return {
                "query": question,
                "answer": "No knowledge base found. Please create one first.",
                "sources": [],
                "num_sources": 0
            }
        
        try:
            # Step 1: Similarity search
            logger.info("Performing similarity search...")
            similar_docs = self.vector_store.similarity_search(question, k=10)
            logger.info(f"Found {len(similar_docs)} similar documents")
            
            if not similar_docs:
                logger.warning("No relevant documents found")
                return {
                    "query": question,
                    "answer": "No relevant information found.",
                    "sources": [],
                    "num_sources": 0
                }
            
            # Step 2: Rerank documents
            logger.info("Reranking documents...")
            reranked_docs = self.reranker.compress_documents(
                documents=similar_docs,
                query=question
            )
            logger.info(f"Reranked to {len(reranked_docs)} documents")
            
            # Step 3: Prepare context
            logger.info("Preparing context for LLM...")
            context = "\n\n".join([doc.page_content for doc in reranked_docs])
            
            # Step 4: Generate response
            logger.info("Generating response...")
            prompt = f"""Use the following context to answer the question. If you cannot answer based on the context, say so clearly.
Context:
{context}
Question: {question}
Answer:"""
            
            response = self.llm.invoke(prompt)
            logger.info("Response generated successfully")
            
            # Prepare sources information
            sources = []
            for i, doc in enumerate(reranked_docs):
                sources.append({
                    "index": i + 1,
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": getattr(doc, 'metadata', {})
                })
            
            result = {
                "query": question,
                "answer": response,
                "sources": sources,
                "num_sources": len(sources)
            }
            
            logger.info(f"Query processed successfully with {len(sources)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise

# ============= GLOBAL RAG INSTANCE =============
rag_instance: Optional[RAGPipeline] = None

# ============= FASTAPI LIFESPAN (Startup/Shutdown) =============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    # Startup
    logger.info("Starting FastAPI application...")
    global rag_instance
    
    try:
        rag_instance = RAGPipeline()
        logger.info("RAG instance created successfully")
    except Exception as e:
        logger.error(f"Failed to create RAG instance: {str(e)}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    rag_instance = None
    logger.info("Cleanup completed")

# ============= DEPENDENCY INJECTION =============
def get_rag_pipeline() -> RAGPipeline:
    """Dependency to get RAG pipeline instance"""
    if rag_instance is None:
        logger.error("RAG pipeline not initialized")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG pipeline not initialized"
        )
    return rag_instance

# ============= FASTAPI APP CREATION =============
app = FastAPI(
    title="RAG API",
    description="Retrieval Augmented Generation API using Gemini and Cohere",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware (allows web browsers to access your API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= API ENDPOINTS =============

@app.get("/", response_model=StatusResponse)
async def root():
    """Health check endpoint"""
    logger.info("Root endpoint accessed")
    return StatusResponse(
        message="RAG API is running",
        status="healthy",
        timestamp=datetime.now().isoformat()
    )

@app.post("/create-knowledge-base", response_model=StatusResponse)
async def create_knowledge_base(
    text_input: TextInput,
    rag: RAGPipeline = Depends(get_rag_pipeline)
):
    """Create a new knowledge base from text"""
    logger.info("Create knowledge base endpoint called")
    
    try:
        rag.create_knowledge_base(text_input.text)
        
        response = StatusResponse(
            message="Knowledge base created successfully",
            status="success",
            timestamp=datetime.now().isoformat(),
            details={"text_length": len(text_input.text)}
        )
        logger.info("Knowledge base created via API")
        return response
        
    except Exception as e:
        logger.error(f"API: Failed to create knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create knowledge base: {str(e)}"
        )

@app.post("/add-text", response_model=StatusResponse)
async def add_text(
    text_input: TextInput,
    rag: RAGPipeline = Depends(get_rag_pipeline)
):
    """Add text to existing knowledge base"""
    logger.info("Add text endpoint called")
    
    try:
        rag.add_text(text_input.text)
        
        response = StatusResponse(
            message="Text added successfully",
            status="success",
            timestamp=datetime.now().isoformat(),
            details={"text_length": len(text_input.text)}
        )
        logger.info("Text added via API")
        return response
        
    except Exception as e:
        logger.error(f"API: Failed to add text: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add text: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_rag(
    query_input: QueryInput,
    rag: RAGPipeline = Depends(get_rag_pipeline)
):
    """Query the RAG system"""
    logger.info(f"Query endpoint called: {query_input.question[:50]}...")
    
    try:
        result = rag.query(query_input.question)
        
        response = QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"],
            num_sources=result["num_sources"],
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Query processed via API with {result['num_sources']} sources")
        return response
        
    except Exception as e:
        logger.error(f"API: Query processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )

@app.get("/status", response_model=StatusResponse)
async def get_status(rag: RAGPipeline = Depends(get_rag_pipeline)):
    """Get system status"""
    logger.info("Status endpoint called")
    
    has_knowledge_base = rag.vector_store is not None
    
    return StatusResponse(
        message="System status retrieved",
        status="success",
        timestamp=datetime.now().isoformat(),
        details={
            "has_knowledge_base": has_knowledge_base,
            "system_ready": True
        }
    )

# ============= RUN THE APPLICATION =============
if __name__ == "__main__":
    logger.info("Starting RAG API server...")
    uvicorn.run(
        "mini_rag:app",  # app_name:variable_name
        host="0.0.0.0",  # Allows external connections
        port=8000,       # Port number
        reload=True,     # Auto-reload on code changes (development only)
        log_level="info"
    )




# # Clean Text Function
# def clean_text(raw_text):
#     # clean_corrupted_text
#     text = raw_text.replace('\n', ' ')  # remove all line breaks
#     text = re.sub(r'\s{2,}', ' ', text)  # remove double spaces
#     text = re.sub(r'([a-z]) ([A-Z])', r'\1. \2', text)  # fix abrupt joins
#     text = re.sub(r'[^a-zA-Z0-9,.:\-–()\'" \n]', '', text)  # remove junk chars
#     text = text.strip()
    
#     # smart_join_lines
#     lines = text.splitlines()
#     cleaned_lines = []
#     buffer = ""
#     for line in lines:
#         stripped = line.strip()
#         if not stripped:
#             continue
#         if buffer and not buffer.endswith(('.', ':', '?', '!', '.”', '”', '*/')):
#             buffer += " " + stripped
#         else:
#             if buffer:
#                 cleaned_lines.append(buffer)
#             buffer = stripped
#     if buffer:
#         cleaned_lines.append(buffer)
#     text = "\n".join(cleaned_lines)
    
#     # clean_ocr_text
#     text = text.replace('\n', ' ')
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'\.\s(?=[a-z])', ' ', text)  # Remove periods not ending sentences
#     text = text.strip()
    
#     # remove_bad_fullstops
#     text = re.sub(r'(?<=[a-zA-Z0-9])\s*\.\s*(?=[a-zA-Z0-9])', ' ', text)
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     # remove_garbage_artifacts
#     text = re.sub(r'\s(?:[a-z]{1,3})(?:\s|$)', ' ', text)
#     text = re.sub(r'\b[^aeiou\s]{4,}\b', '', text)
#     text = re.sub(r'[^\w\s]{2,}', ' ', text)
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     return text


# #Vector DB
# def build_vector_store(user_text, use_pinecone=False):
    
#     # Split into chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1200,  # Small chunks for precise retrieval
#         chunk_overlap=120  # 10% Overlap to avoid losing context
#     )
#     chunks = text_splitter.create_documents([user_text])
    
#     # Embeddings model
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     # if use_pinecone:
#     #     # Pinecone setup (uncomment when ready)
#     #     # pinecone.init(api_key="your-pinecone-api-key", environment="your-env")
#     #     # index_name = "your-index-name"
#     #     # vectorstore = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
#     #     pass
#     # else:
#     #     
#     # # FAISS: Local, in-memory
#     vectorstore = FAISS.from_documents(chunks, embeddings)
    
#     return vectorstore, chunks

# # Step 2: Create retriever and reranker
# def create_retriever_and_reranker(vectorstore):
#     # Retriever: Fetches top-k chunks based on similarity
#     retriever = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 5}  # Return top 5 documents
# ) 
    
#     # Reranker: Re-scores retrieved chunks for better relevance
#     reranker = CohereRerank(
#         model="rerank-english-v3.0",  # Latest Cohere rerank model
#         top_n=5,                      # Return top 5 after reranking
#         cohere_api_key=os.getenv("COHERE_API_KEY")  # Set your API key
#     )
    
#     return retriever, reranker

# # Step 3: Create RAG chain with retriever and reranker
# def create_rag_chain(vectorstore, retriever, reranker):
#     llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    
#     # Custom prompt for better answers
#     # prompt = [
#     #     SystemMessage(content='You are a great assistant. Please answer the question ONLY FROM THE provided context. If the context seems insufficient, reply "The context seems insufficient to answer the question."'),
#     #     HumanMessage(content="Context: {context} Question: {query} Answer:")
#     # ]
#     # prompt = PromptTemplate.from_messages(prompt_template)

#         # Create proper PromptTemplate for RetrievalQA
#     prompt_template = """You are a great assistant. Please answer the question ONLY FROM THE provided context. If the context seems insufficient, reply "The context seems insufficient to answer the question."

# Context: {context}

# Question: {question}

# Answer:"""
    
#     prompt = PromptTemplate(
#         template=prompt_template,
#         input_variables=["context", "question"]
#     )
    
#     # Custom retrieval function with reranking
#     def rerank_and_format_docs(query, retriever, reranker, top_k=3):
#         # Step 1: Retrieve initial documents
#         docs = retriever.get_relevant_documents(query)
#         if not docs:
#             return []
        
#         # Step 2: Rerank documents
#         pairs = [[query, doc.page_content] for doc in docs]
#         scores = reranker.predict(pairs)  # Get relevance scores
#         ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
#         ranked_docs = [docs[i] for i in ranked_indices[:top_k]]  # Select top_k
        
#         return ranked_docs
    
#     # Create RetrievalQA chain
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,  # Initial retriever
#         return_source_documents=True,
#         chain_type_kwargs={
#             "prompt": prompt,
#             "document_prompt": PromptTemplate(
#                 input_variables=["page_content"],
#                 template="{page_content}"
#             ),
#         }
#     )
    
#     return qa_chain

# # Main function to run the RAG
# # def run_rag():
# #     # Get user text (from document pattern)
# #     user_text = input("Enter your text: ").strip()
# #     if not user_text:
# #         print("No text provided. Exiting.")
# #         return
    
# #     # Clean text
# #     cleaned_text = clean_text(user_text)
# #     print("\nCleaned text:", cleaned_text)
    
# #     # Build vector store
# #     vectorstore, chunks = build_vector_store(cleaned_text)
# #     print("\nNumber of chunks:", len(chunks))
    
# #     # Create retriever and reranker
# #     retriever, reranker = create_retriever_and_reranker(vectorstore)
    
# #     # Create RAG chain
# #     rag_chain = create_rag_chain(vectorstore, retriever, reranker)
    
# #     # Query loop
# #     while True:
# #         query = input("\nEnter your question (or 'exit' to quit): ").strip()
# #         if query.lower() == 'exit':
# #             break
# #         result = rag_chain.invoke({"query": query})
# #         print("\nAnswer:", result['result'])
# #         if 'source_documents' in result:
# #             print("\nSources:", [doc.page_content[:100] + "..." for doc in result['source_documents']])


# def run_rag():
#     # Get user text
#     user_text = input("Enter your text: ").strip()
#     if not user_text:
#         print("No text provided. Exiting.")
#         return
    
#     # Clean text
#     cleaned_text = clean_text(user_text)
#     print(f"\nCleaned text: {cleaned_text[:100]}..." if len(cleaned_text) > 100 else f"\nCleaned text: {cleaned_text}")
    
#     # Build vector store
#     vectorstore, chunks = build_vector_store(cleaned_text)
#     print(f"\nNumber of chunks: {len(chunks)}")
    
#     # Create retriever and reranker
#     retriever, reranker = create_retriever_and_reranker(vectorstore)
    
#     # Create RAG chain (choose one approach)
#     rag_chain = create_rag_chain(vectorstore, retriever, reranker)  # Simple approach
#     # OR
#     # rag_chain = create_rag_chain_with_reranking(vectorstore, retriever, reranker)  # With reranking
    
#     # Query loop
#     while True:
#         query = input("\nEnter your question (or 'exit' to quit): ").strip()
#         if query.lower() == 'exit':
#             break
        
#         try:
#             result = rag_chain.invoke({"query": query})
#             print("\nAnswer:", result['result'])
            
#             if 'source_documents' in result and result['source_documents']:
#                 print("\nSources:")
#                 for i, doc in enumerate(result['source_documents'], 1):
#                     print(f"{i}. {doc.page_content[:100]}...")
#         except Exception as e:
#             print(f"Error: {e}")
# # Run the main function
# if __name__ == "__main__":
#     run_rag()