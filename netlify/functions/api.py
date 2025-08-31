import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import your existing RAG components
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_cohere import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Set environment variables
os.environ["GOOGLE_API_KEY"] = "AIzaSyCITsocNmlmc2WE2EzvffSgrGElX153Nak"
os.environ["COHERE_API_KEY"] = "AejGUl4A1TosuUZFoeyiNixW65zEvNNUbeB3dFkf"

# Global RAG instance (persists between function calls)
rag_instance = None

class RAGPipeline:
    def __init__(self):
        self.embeddings, self.llm, self.text_splitter, self.reranker = self._setup_components()
        self.vector_store = None
    
    def _setup_components(self):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
        )
        reranker = CohereRerank(model="rerank-english-v3.0", top_n=5)
        return embeddings, llm, text_splitter, reranker
    
    def create_knowledge_base(self, text_input: str):
        docs = self.text_splitter.create_documents([text_input])
        self.vector_store = FAISS.from_documents(documents=docs, embedding=self.embeddings)
        return self.vector_store
    
    def add_text(self, new_text: str):
        if self.vector_store is None:
            return self.create_knowledge_base(new_text)
        new_docs = self.text_splitter.create_documents([new_text])
        self.vector_store.add_documents(new_docs)
        return self.vector_store
    
    def query(self, question: str) -> Dict[str, Any]:
        if self.vector_store is None:
            return {
                "query": question,
                "answer": "No knowledge base found. Please create one first.",
                "sources": [],
                "num_sources": 0
            }
        
        similar_docs = self.vector_store.similarity_search(question, k=10)
        if not similar_docs:
            return {
                "query": question,
                "answer": "No relevant information found.",
                "sources": [],
                "num_sources": 0
            }
        
        reranked_docs = self.reranker.compress_documents(documents=similar_docs, query=question)
        context = "\n\n".join([doc.page_content for doc in reranked_docs])
        
        prompt = f"""Use the following context to answer the question. If you cannot answer based on the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.llm.invoke(prompt)
        
        sources = []
        for i, doc in enumerate(reranked_docs):
            sources.append({
                "index": i + 1,
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "metadata": getattr(doc, 'metadata', {})
            })
        
        return {
            "query": question,
            "answer": response,
            "sources": sources,
            "num_sources": len(sources)
        }

def handler(event, context):
    global rag_instance
    
    # Initialize RAG if not exists
    if rag_instance is None:
        rag_instance = RAGPipeline()
    
    try:
        # Parse the request
        path = event.get('path', '')
        method = event.get('httpMethod', 'GET')
        body = event.get('body', '{}')
        
        if body:
            data = json.loads(body)
        else:
            data = {}
        
        # Handle different endpoints
        if method == 'GET' and path.endswith('/'):
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'message': 'RAG API is running',
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat()
                })
            }
        
        elif method == 'POST' and 'create-knowledge-base' in path:
            text = data.get('text', '')
            rag_instance.create_knowledge_base(text)
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'message': 'Knowledge base created successfully',
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                })
            }
        
        elif method == 'POST' and 'add-text' in path:
            text = data.get('text', '')
            rag_instance.add_text(text)
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'message': 'Text added successfully',
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                })
            }
        
        elif method == 'POST' and 'query' in path:
            question = data.get('question', '')
            result = rag_instance.query(question)
            result['timestamp'] = datetime.now().isoformat()
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(result)
            }
        
        # Handle OPTIONS for CORS
        elif method == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                'body': ''
            }
        
        else:
            return {
                'statusCode': 404,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'Endpoint not found'})
            }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }
