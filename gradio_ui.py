import gradio as gr
import requests
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Backend API URL - adjust this to match your FastAPI server
API_BASE_URL = "http://localhost:8000"

class RAGGradioInterface:
    def __init__(self, api_base_url: str = API_BASE_URL):
        self.api_base_url = api_base_url
        
    def check_api_status(self) -> tuple[str, str]:
        """Check if the API is running"""
        try:
            response = requests.get(f"{self.api_base_url}/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                status = "🟢 API Connected"
                details = f"Knowledge Base: {'✅ Ready' if data.get('details', {}).get('has_knowledge_base', False) else '❌ Empty'}"
                return status, details
            else:
                return "🔴 API Error", f"Status Code: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return "🔴 API Disconnected", f"Error: {str(e)}"
    
    def create_knowledge_base(self, text: str) -> tuple[str, str]:
        """Create a new knowledge base"""
        if not text.strip():
            return "❌ Error: Please enter some text", ""
        
        try:
            response = requests.post(
                f"{self.api_base_url}/create-knowledge-base",
                json={"text": text},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                success_msg = f"✅ Knowledge base created successfully!\nText length: {data.get('details', {}).get('text_length', 'N/A')} characters"
                return success_msg, ""
            else:
                error_data = response.json()
                return f"❌ Error: {error_data.get('detail', 'Unknown error')}", ""
                
        except requests.exceptions.RequestException as e:
            return f"❌ Connection Error: {str(e)}", ""
    
    def add_text(self, text: str, current_status: str) -> tuple[str, str]:
        """Add text to existing knowledge base"""
        if not text.strip():
            return "❌ Error: Please enter some text", current_status
        
        try:
            response = requests.post(
                f"{self.api_base_url}/add-text",
                json={"text": text},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                success_msg = f"✅ Text added successfully!\nText length: {data.get('details', {}).get('text_length', 'N/A')} characters"
                return success_msg, ""
            else:
                error_data = response.json()
                return f"❌ Error: {error_data.get('detail', 'Unknown error')}", current_status
                
        except requests.exceptions.RequestException as e:
            return f"❌ Connection Error: {str(e)}", current_status
    
    def query_rag(self, question: str, history: list) -> tuple[list, str]:
        """Query the RAG system"""
        if not question.strip():
            return history, ""
        
        # Add user message to history
        history.append({"role": "user", "content": question})
        
        try:
            response = requests.post(
                f"{self.api_base_url}/query",
                json={"question": question},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer received")
                sources = data.get("sources", [])
                
                # Format response with sources
                formatted_answer = f"{answer}\n\n"
                if sources:
                    formatted_answer += f"**Sources ({len(sources)} found):**\n"
                    for i, source in enumerate(sources, 1):
                        content_preview = source.get("content", "")[:150] + "..."
                        formatted_answer += f"{i}. {content_preview}\n"
                
                # Add assistant response to history
                history.append({"role": "assistant", "content": formatted_answer})
                
            else:
                error_data = response.json()
                error_msg = f"❌ Error: {error_data.get('detail', 'Unknown error')}"
                history.append({"role": "assistant", "content": error_msg})
                
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ Connection Error: {str(e)}"
            history.append({"role": "assistant", "content": error_msg})
        
        return history, ""

# Create the interface instance
rag_interface = RAGGradioInterface()

# Define CSS for better styling
css = """
.main-container {
    max-width: 1200px;
    margin: 0 auto;
}
.status-box {
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.success {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}
.error {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}
"""

# Create the Gradio interface
with gr.Blocks(css=css, title="RAG System Interface") as demo:
    gr.Markdown("""
    # 🤖 RAG System Interface
    
    **Retrieval Augmented Generation** system powered by Gemini and Cohere
    
    Use this interface to:
    - 📚 Create and manage your knowledge base
    - 💬 Query the RAG system with intelligent responses
    - 📊 Monitor system status
    """)
    
    with gr.Tab("📊 System Status"):
        with gr.Row():
            status_button = gr.Button("🔄 Check Status", variant="primary")
            
        with gr.Row():
            api_status = gr.Textbox(label="API Status", interactive=False)
            kb_status = gr.Textbox(label="Knowledge Base Status", interactive=False)
    
    with gr.Tab("📚 Knowledge Base Management"):
        gr.Markdown("### Create or Add to Knowledge Base")
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text Content",
                    placeholder="Enter text to add to your knowledge base...",
                    lines=8,
                    max_lines=15
                )
                
                with gr.Row():
                    create_kb_btn = gr.Button("🆕 Create New Knowledge Base", variant="primary")
                    add_text_btn = gr.Button("➕ Add to Existing Knowledge Base", variant="secondary")
                
            with gr.Column():
                kb_result = gr.Textbox(
                    label="Result",
                    lines=6,
                    interactive=False
                )
    
    with gr.Tab("💬 Chat & Query"):
        gr.Markdown("### Ask Questions About Your Knowledge Base")
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="RAG Assistant",
                    height=500,
                    type="messages",
                    placeholder="Your conversation will appear here..."
                )
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about your knowledge base...",
                        lines=2,
                        scale=4
                    )
                    query_btn = gr.Button("🚀 Ask", variant="primary", scale=1)
                
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("""
                ### 💡 Tips:
                - Make sure you have created a knowledge base first
                - Ask specific questions for better results
                - The system will show relevant sources
                - Responses include context from your documents
                """)
    
    # Event handlers
    status_button.click(
        fn=rag_interface.check_api_status,
        outputs=[api_status, kb_status]
    )
    
    create_kb_btn.click(
        fn=rag_interface.create_knowledge_base,
        inputs=[text_input],
        outputs=[kb_result, text_input]
    )
    
    add_text_btn.click(
        fn=rag_interface.add_text,
        inputs=[text_input, kb_result],
        outputs=[kb_result, text_input]
    )
    
    # Query handlers
    def handle_query(question, history):
        return rag_interface.query_rag(question, history)
    
    query_btn.click(
        fn=handle_query,
        inputs=[query_input, chatbot],
        outputs=[chatbot, query_input]
    )
    
    query_input.submit(
        fn=handle_query,
        inputs=[query_input, chatbot],
        outputs=[chatbot, query_input]
    )
    
    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, query_input]
    )
    
    # Load initial status on startup
    demo.load(
        fn=rag_interface.check_api_status,
        outputs=[api_status, kb_status]
    )

# Launch configuration
if __name__ == "__main__":
    print("🚀 Starting RAG Gradio Interface...")
    print(f"📡 Connecting to API at: {API_BASE_URL}")
    print("🌐 Gradio interface will be available at: http://localhost:7860")
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Gradio default port
        share=False,            # Set to True to create public link
        debug=True,             # Enable debug mode
        show_error=True         # Show detailed error messages
    )


# import gradio as gr
# import requests
# import json
# from datetime import datetime
# from typing import Dict, Any, Optional, List, Tuple
# import time

# # Backend API URL - adjust this to match your FastAPI server
# API_BASE_URL = "http://localhost:8000"

# class EnhancedRAGInterface:
#     def __init__(self, api_base_url: str = API_BASE_URL):
#         self.api_base_url = api_base_url
#         self.uploaded_texts = []
#         self.total_tokens_used = 0
#         self.total_cost = 0.0
        
#     def estimate_tokens(self, text: str) -> int:
#         """More accurate token estimation"""
#         if not text:
#             return 0
#         # Better estimation: ~3.8 characters per token on average
#         return max(1, int(len(text) / 3.8))
    
#     def estimate_cost(self, tokens: int) -> float:
#         """Cost estimation based on current API pricing"""
#         # Gemini pricing: roughly $0.000001 per token
#         return tokens * 0.000001
    
#     def format_currency(self, amount: float) -> str:
#         """Format currency with proper precision"""
#         return f"${amount:.6f}"
    
#     def get_detailed_stats(self, text: str) -> str:
#         """Get comprehensive text statistics"""
#         if not text.strip():
#             return ""
        
#         chars = len(text)
#         words = len(text.split())
#         lines = text.count('\n') + 1
#         tokens = self.estimate_tokens(text)
#         cost = self.estimate_cost(tokens)
        
#         return f"""
#         📊 **Text Analysis:**
#         • **Characters:** {chars:,}
#         • **Words:** {words:,}  
#         • **Lines:** {lines:,}
#         • **Estimated Tokens:** ~{tokens:,}
#         • **Estimated Cost:** {self.format_currency(cost)}
#         """
    
#     def check_api_health(self) -> Tuple[str, bool]:
#         """Check if API is responsive"""
#         try:
#             response = requests.get(f"{self.api_base_url}/status", timeout=5)
#             if response.status_code == 200:
#                 data = response.json()
#                 has_kb = data.get('details', {}).get('has_knowledge_base', False)
#                 status = f"🟢 **API Connected** | Knowledge Base: {'✅ Active' if has_kb else '❌ Empty'}"
#                 return status, has_kb
#             else:
#                 return f"🟡 **API Warning** | Status: {response.status_code}", False
#         except Exception as e:
#             return f"🔴 **API Disconnected** | {str(e)[:50]}...", False
    
#     def add_text_with_progress(self, text: str, has_kb: bool) -> Tuple[str, str, bool, str, str, str]:
#         """Enhanced text addition with better feedback"""
#         if not text.strip():
#             return (
#                 "⚠️ **Error:** Please enter some text first!",
#                 text,
#                 has_kb,
#                 self.format_uploaded_content(),
#                 "",
#                 "❌ No text provided"
#             )
        
#         # Validate text length
#         if len(text) < 10:
#             return (
#                 "⚠️ **Warning:** Text is very short. Consider adding more content for better results.",
#                 text,
#                 has_kb,
#                 self.format_uploaded_content(),
#                 "",
#                 "⚠️ Text too short"
#             )
        
#         try:
#             start_time = time.time()
            
#             endpoint = '/add-text' if has_kb else '/create-knowledge-base'
#             action = "Adding" if has_kb else "Creating knowledge base"
            
#             response = requests.post(
#                 f"{self.api_base_url}{endpoint}",
#                 json={"text": text},
#                 timeout=60  # Longer timeout for large texts
#             )
            
#             end_time = time.time()
#             duration = end_time - start_time
            
#             if response.status_code == 200:
#                 # Track the uploaded text
#                 text_entry = {
#                     'id': len(self.uploaded_texts) + 1,
#                     'text': text,
#                     'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     'chars': len(text),
#                     'tokens': self.estimate_tokens(text),
#                     'cost': self.estimate_cost(self.estimate_tokens(text))
#                 }
#                 self.uploaded_texts.append(text_entry)
                
#                 # Update totals
#                 self.total_tokens_used += text_entry['tokens']
#                 self.total_cost += text_entry['cost']
                
#                 success_msg = f"""
#                 ✅ **Success!** Text added to knowledge base
                
#                 📈 **Performance:**
#                 • Processing Time: {duration:.2f}s
#                 • Text Length: {len(text):,} characters
#                 • Estimated Tokens: {text_entry['tokens']:,}
#                 • Cost: {self.format_currency(text_entry['cost'])}
                
#                 🎯 **Status:** {'Knowledge base updated' if has_kb else 'Knowledge base created'}
#                 """
                
#                 timing_msg = f"⏱️ **Last Operation:** {duration:.2f} seconds"
#                 status_msg = "✅ Ready to answer questions!"
                
#                 return (
#                     success_msg,
#                     "",  # Clear input
#                     True,  # Has KB now
#                     self.format_uploaded_content(),
#                     timing_msg,
#                     status_msg
#                 )
#             else:
#                 error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
#                 error_msg = f"""
#                 ❌ **Error:** {error_data.get('detail', 'Unknown server error')}
                
#                 🔍 **Debug Info:**
#                 • Status Code: {response.status_code}
#                 • Duration: {duration:.2f}s
#                 • Endpoint: {endpoint}
#                 """
#                 return (
#                     error_msg,
#                     text,
#                     has_kb,
#                     self.format_uploaded_content(),
#                     f"❌ Failed in {duration:.2f}s",
#                     "❌ Operation failed"
#                 )
                
#         except requests.exceptions.Timeout:
#             return (
#                 "⏰ **Timeout Error:** Request took too long. Try with smaller text or check your connection.",
#                 text,
#                 has_kb,
#                 self.format_uploaded_content(),
#                 "❌ Request timeout",
#                 "⏰ Operation timed out"
#             )
#         except Exception as e:
#             return (
#                 f"🔌 **Connection Error:** {str(e)}",
#                 text,
#                 has_kb,
#                 self.format_uploaded_content(),
#                 "❌ Connection failed",
#                 "🔌 API unreachable"
#             )
    
#     def query_with_enhanced_response(self, question: str, has_kb: bool, history: List) -> Tuple[List, str, str, str, str]:
#         """Enhanced querying with rich response formatting"""
#         if not question.strip():
#             return history, "", "⚠️ Please enter a question!", "", ""
        
#         if not has_kb:
#             warning_msg = """
#             ❌ **No Knowledge Base Found**
            
#             Please add some text to create a knowledge base first!
            
#             💡 **Quick Start:**
#             1. Go to the "Knowledge Base" section above
#             2. Paste or type your content
#             3. Click "Create Knowledge Base"
#             4. Come back here to ask questions!
#             """
#             return history, question, warning_msg, "", ""
        
#         try:
#             start_time = time.time()
            
#             response = requests.post(
#                 f"{self.api_base_url}/query",
#                 json={"question": question},
#                 timeout=60
#             )
            
#             end_time = time.time()
#             duration = end_time - start_time
            
#             if response.status_code == 200:
#                 data = response.json()
#                 answer = data.get("answer", "No answer received")
#                 sources = data.get("sources", [])
                
#                 # Calculate tokens and cost
#                 total_text = question + answer
#                 tokens_used = self.estimate_tokens(total_text)
#                 query_cost = self.estimate_cost(tokens_used)
#                 self.total_tokens_used += tokens_used
#                 self.total_cost += query_cost
                
#                 # Add to history with enhanced formatting
#                 formatted_answer = f"""
#                 {answer}
                
#                 ---
                
#                 📊 **Query Stats:** {len(sources)} sources • {tokens_used:,} tokens • {self.format_currency(query_cost)} cost • {duration:.2f}s response time
#                 """
                
#                 history.append([question, formatted_answer])
                
#                 # Create detailed metadata
#                 metadata = f"""
#                 🎯 **Query Performance:**
#                 • **Response Time:** {duration:.2f} seconds
#                 • **Sources Found:** {len(sources)}
#                 • **Tokens Used:** {tokens_used:,}
#                 • **Query Cost:** {self.format_currency(query_cost)}
#                 • **Processed At:** {datetime.now().strftime("%H:%M:%S")}
                
#                 📈 **Session Totals:**
#                 • **Total Tokens:** {self.total_tokens_used:,}
#                 • **Total Cost:** {self.format_currency(self.total_cost)}
#                 """
                
#                 # Format sources
#                 sources_text = ""
#                 if sources:
#                     sources_text = f"## 📚 Sources & Citations ({len(sources)} found)\n\n"
#                     for i, source in enumerate(sources, 1):
#                         content_preview = source.get('content', '')[:200]
#                         if len(source.get('content', '')) > 200:
#                             content_preview += "..."
                        
#                         sources_text += f"""
#                         **📄 Source {i}:**
#                         {content_preview}
                        
#                         ---
#                         """
#                 else:
#                     sources_text = "ℹ️ No specific sources were cited for this response."
                
#                 timing_msg = f"⚡ **Last Query:** {duration:.2f} seconds"
                
#                 return (
#                     history, 
#                     "",  # Clear question input
#                     "✅ Question answered successfully!",
#                     metadata,
#                     timing_msg,
#                     sources_text
#                 )
#             else:
#                 error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
#                 error_msg = f"❌ **Query Failed:** {error_data.get('detail', 'Unknown error')}"
#                 history.append([question, error_msg])
#                 return (
#                     history, 
#                     question,
#                     error_msg,
#                     f"❌ **Error Response Time:** {duration:.2f}s",
#                     f"❌ Failed in {duration:.2f}s"
#                 )
                
#         except Exception as e:
#             error_msg = f"🔌 **Connection Error:** {str(e)}"
#             history.append([question, error_msg])
#             return (
#                 history, 
#                 question,
#                 error_msg,
#                 "",
#                 "❌ Connection failed"
#             )
    
#     def format_uploaded_content(self) -> str:
#         """Format uploaded content with rich details"""
#         if not self.uploaded_texts:
#             return """
#             📝 **Knowledge Base Status:** Empty
            
#             ℹ️ Add some text above to get started!
#             """
        
#         content = f"""
#         ## 📚 Knowledge Base Content ({len(self.uploaded_texts)} entries)
        
#         **📊 Summary:**
#         • **Total Entries:** {len(self.uploaded_texts)}
#         • **Total Characters:** {sum(item['chars'] for item in self.uploaded_texts):,}
#         • **Estimated Tokens:** {sum(item['tokens'] for item in self.uploaded_texts):,}
#         • **Estimated Setup Cost:** {self.format_currency(sum(item['cost'] for item in self.uploaded_texts))}
        
#         ---
        
#         """
        
#         for i, item in enumerate(self.uploaded_texts, 1):
#             preview = item['text'][:150]
#             if len(item['text']) > 150:
#                 preview += "..."
            
#             content += f"""
#             **📄 Entry #{i}** • Added: {item['timestamp']}
            
#             {preview}
            
#             *📈 Stats: {item['chars']:,} chars • {item['tokens']:,} tokens • {self.format_currency(item['cost'])} cost*
            
#             ---
            
#             """
        
#         return content
    
#     def clear_all_data(self) -> Tuple[str, str, bool, str, str]:
#         """Clear all data and reset interface"""
#         self.uploaded_texts = []
#         self.total_tokens_used = 0
#         self.total_cost = 0.0
        
#         return (
#             "🗑️ **All data cleared!** You can start fresh now.",
#             "",
#             False,
#             "📝 **Knowledge Base Status:** Empty\n\nℹ️ Add some text to get started!",
#             ""
#         )

# # Initialize the enhanced interface
# rag_interface = EnhancedRAGInterface()

# # Premium CSS styling
# premium_css = """
# /* Import Google Fonts */
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

# /* Global Styling */
# .gradio-container {
#     max-width: 1400px !important;
#     margin: 0 auto !important;
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%) !important;
#     min-height: 100vh;
#     font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
#     padding: 20px;
# }

# /* Header Styling */
# .hero-header {
#     background: linear-gradient(45deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 50%, rgba(240, 147, 251, 0.95) 100%) !important;
#     backdrop-filter: blur(20px) !important;
#     color: white !important;
#     padding: 40px !important;
#     border-radius: 25px !important;
#     margin-bottom: 30px !important;
#     box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4) !important;
#     border: 1px solid rgba(255, 255, 255, 0.2) !important;
#     text-align: center !important;
# }

# .hero-title {
#     font-size: 3.5em !important;
#     font-weight: 800 !important;
#     margin: 0 0 20px 0 !important;
#     text-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
#     background: linear-gradient(45deg, #ffffff, #f0f8ff) !important;
#     -webkit-background-clip: text !important;
#     -webkit-text-fill-color: transparent !important;
#     background-clip: text !important;
# }

# .hero-subtitle {
#     font-size: 1.3em !important;
#     margin: 10px 0 !important;
#     opacity: 0.95 !important;
#     font-weight: 500 !important;
# }

# .hero-notice {
#     font-size: 1em !important;
#     margin: 5px 0 !important;
#     opacity: 0.9 !important;
#     background: rgba(255, 255, 255, 0.1) !important;
#     padding: 10px 20px !important;
#     border-radius: 15px !important;
#     display: inline-block !important;
# }

# /* Section Cards */
# .section-card {
#     background: rgba(255, 255, 255, 0.98) !important;
#     border-radius: 25px !important;
#     padding: 35px !important;
#     margin-bottom: 30px !important;
#     box-shadow: 0 15px 50px rgba(0,0,0,0.1) !important;
#     border: 1px solid rgba(255, 255, 255, 0.3) !important;
#     backdrop-filter: blur(10px) !important;
#     transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
# }

# .section-card:hover {
#     transform: translateY(-8px) !important;
#     box-shadow: 0 25px 70px rgba(0,0,0,0.15) !important;
# }

# /* Section Headers */
# .section-header {
#     font-size: 1.8em !important;
#     font-weight: 700 !important;
#     margin-bottom: 25px !important;
#     background: linear-gradient(135deg, #667eea, #764ba2) !important;
#     -webkit-background-clip: text !important;
#     -webkit-text-fill-color: transparent !important;
#     background-clip: text !important;
#     display: flex !important;
#     align-items: center !important;
#     gap: 10px !important;
# }

# /* Input Styling */
# .gradio-textbox, .gradio-textarea {
#     border: 3px solid transparent !important;
#     border-radius: 20px !important;
#     background: linear-gradient(white, white) padding-box, linear-gradient(135deg, #667eea, #764ba2) border-box !important;
#     font-size: 16px !important;
#     font-family: 'Inter', sans-serif !important;
#     transition: all 0.3s ease !important;
#     padding: 18px !important;
# }

# .gradio-textbox:focus, .gradio-textarea:focus {
#     transform: scale(1.02) !important;
#     box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3) !important;
#     outline: none !important;
# }

# /* Button Styling */
# .gradio-button {
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
#     color: white !important;
#     border: none !important;
#     border-radius: 25px !important;
#     font-size: 16px !important;
#     font-weight: 700 !important;
#     padding: 18px 35px !important;
#     text-transform: uppercase !important;
#     letter-spacing: 1px !important;
#     transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
#     box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
#     cursor: pointer !important;
# }

# .gradio-button:hover {
#     transform: translateY(-3px) scale(1.05) !important;
#     box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6) !important;
#     background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important;
# }

# .gradio-button:active {
#     transform: translateY(0px) scale(0.98) !important;
# }

# /* Status and Info Displays */
# .status-success {
#     background: linear-gradient(135deg, #d4edda, #c3e6cb) !important;
#     color: #155724 !important;
#     border: 2px solid #48bb78 !important;
#     border-radius: 15px !important;
#     padding: 20px !important;
#     margin: 15px 0 !important;
# }

# .status-error {
#     background: linear-gradient(135deg, #f8d7da, #f5c6cb) !important;
#     color: #721c24 !important;
#     border: 2px solid #dc3545 !important;
#     border-radius: 15px !important;
#     padding: 20px !important;
#     margin: 15px 0 !important;
# }

# .status-warning {
#     background: linear-gradient(135deg, #fff3cd, #ffeaa7) !important;
#     color: #856404 !important;
#     border: 2px solid #ffc107 !important;
#     border-radius: 15px !important;
#     padding: 20px !important;
#     margin: 15px 0 !important;
# }

# .timing-display {
#     background: linear-gradient(135deg, #e6fffa, #b2f5ea) !important;
#     color: #234e52 !important;
#     border: 3px solid #4fd1c7 !important;
#     border-radius: 20px !important;
#     padding: 25px !important;
#     text-align: center !important;
#     font-weight: 600 !important;
#     font-size: 18px !important;
#     margin: 20px 0 !important;
#     box-shadow: 0 8px 25px rgba(79, 209, 199, 0.3) !important;
# }

# /* Chat Interface */
# .gradio-chatbot {
#     border-radius: 20px !important;
#     border: 3px solid rgba(102, 126, 234, 0.3) !important;
#     box-shadow: inset 0 4px 15px rgba(0,0,0,0.05) !important;
#     background: linear-gradient(135deg, #f8fafc, #ffffff) !important;
# }

# /* Stats Display */
# .stats-card {
#     background: linear-gradient(135deg, #f7fafc, #edf2f7) !important;
#     border: 2px solid #e2e8f0 !important;
#     border-left: 8px solid #667eea !important;
#     border-radius: 15px !important;
#     padding: 20px !important;
#     margin: 15px 0 !important;
#     font-family: 'Monaco', 'Menlo', monospace !important;
#     font-size: 14px !important;
# }

# /* Responsive Design */
# @media (max-width: 768px) {
#     .gradio-container {
#         padding: 10px !important;
#     }
    
#     .hero-title {
#         font-size: 2.5em !important;
#     }
    
#     .section-card {
#         padding: 20px !important;
#     }
# }

# /* Animations */
# @keyframes fadeInUp {
#     from {
#         opacity: 0;
#         transform: translateY(30px);
#     }
#     to {
#         opacity: 1;
#         transform: translateY(0);
#     }
# }

# .section-card {
#     animation: fadeInUp 0.6s ease-out !important;
# }

# /* Scrollbar Styling */
# ::-webkit-scrollbar {
#     width: 12px;
# }

# ::-webkit-scrollbar-track {
#     background: rgba(255, 255, 255, 0.1);
#     border-radius: 10px;
# }

# ::-webkit-scrollbar-thumb {
#     background: linear-gradient(135deg, #667eea, #764ba2);
#     border-radius: 10px;
# }

# ::-webkit-scrollbar-thumb:hover {
#     background: linear-gradient(135deg, #5a67d8, #6b46c1);
# }
# """

# # Create the enhanced Gradio interface
# with gr.Blocks(css=premium_css, title="🤖 Advanced RAG Chat System", theme=gr.themes.Soft()) as demo:
    
#     # Hero Header
#     gr.HTML("""
#     <div class="hero-header">
#         <h1 class="hero-title">🤖 Advanced RAG Chat System</h1>
#         <p class="hero-subtitle">Powered by Gemini AI & Cohere Reranking • Enterprise-Grade RAG Pipeline</p>
#         <p class="hero-notice"><b>💡 Pro Tip:</b> Add comprehensive text for better AI responses • Scroll down to see detailed results</p>
#     </div>
#     """)
    
#     # State management
#     has_knowledge_base = gr.State(False)
    
#     # API Health Check Section
#     with gr.Group(elem_classes=["section-card"]):
#         gr.HTML('<h2 class="section-header">🔍 System Health Monitor</h2>')
        
#         with gr.Row():
#             health_check_btn = gr.Button("🔄 Check API Status", variant="primary", scale=1)
#             clear_all_btn = gr.Button("🗑️ Clear All Data", variant="stop", scale=1)
        
#         api_health_status = gr.Markdown("🔄 Click 'Check API Status' to verify connection...")
    
#     # Knowledge Base Management Section  
#     with gr.Group(elem_classes=["section-card"]):
#         gr.HTML('<h2 class="section-header">📚 Knowledge Base Management</h2>')
        
#         with gr.Row():
#             with gr.Column(scale=3):
#                 text_input = gr.Textbox(
#                     label="📝 Document Content",
#                     placeholder="Paste your documents, articles, or any text content here...\n\nTip: Longer, detailed text produces better AI responses!",
#                     lines=8,
#                     max_lines=15,
#                     show_copy_button=True
#                 )
                
#                 # Real-time statistics
#                 text_stats_display = gr.Markdown("", elem_classes=["stats-card"])
                
#             with gr.Column(scale=1):
#                 add_text_btn = gr.Button(
#                     "🚀 Create Knowledge Base", 
#                     variant="primary", 
#                     size="lg",
#                     scale=1
#                 )
                
#                 gr.HTML("""
#                 <div style="background: linear-gradient(135deg, #e6fffa, #b2f5ea); padding: 15px; border-radius: 10px; margin-top: 15px;">
#                     <h4>💡 Quick Tips:</h4>
#                     <ul style="font-size: 13px; margin: 10px 0;">
#                         <li>📄 Add multiple documents for richer context</li>
#                         <li>📊 Longer texts = better AI understanding</li>
#                         <li>🔄 You can add more content anytime</li>
#                         <li>💰 Cost tracking included</li>
#                     </ul>
#                 </div>
#                 """)
        
#         # Status and results
#         kb_status_display = gr.Markdown("", elem_classes=["status-success"])
        
#         # Uploaded content display
#         uploaded_content_display = gr.Markdown(
#             "📝 **Knowledge Base Status:** Empty\n\nℹ️ Add some text to get started!",
#             elem_classes=["stats-card"]
#         )
    
#     # Q&A Section
#     with gr.Group(elem_classes=["section-card"]):
#         gr.HTML('<h2 class="section-header">💬 Intelligent Q&A Interface</h2>')
        
#         with gr.Row():
#             with gr.Column(scale=4):
#                 question_input = gr.Textbox(
#                     label="🤔 Your Question",
#                     placeholder="Ask anything about your uploaded content... Be specific for better results!",
#                     lines=2,
#                     show_copy_button=True
#                 )
                
#                 with gr.Row():
#                     ask_btn = gr.Button("🚀 Get Answer", variant="primary", scale=2)
#                     clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)
                
#                 question_status = gr.Markdown("", elem_classes=["status-success"])
                
#             with gr.Column(scale=1):
#                 gr.HTML("""
#                 <div style="background: linear-gradient(135deg, #fff3cd, #ffeaa7); padding: 15px; border-radius: 10px;">
#                     <h4>🎯 Question Tips:</h4>
#                     <ul style="font-size: 13px; margin: 10px 0;">
#                         <li>🔍 Be specific and detailed</li>
#                         <li>📋 Ask for summaries, lists, or analysis</li>
#                         <li>🔗 Reference specific topics from your content</li>
#                         <li>💡 Try follow-up questions</li>
#                     </ul>
#                 </div>
#                 """)
    
#     # Performance Metrics
#     with gr.Group(elem_classes=["section-card"]):
#         gr.HTML('<h2 class="section-header">⚡ Performance Dashboard</h2>')
        
#         with gr.Row():
#             timing_display = gr.HTML("""
#             <div class="timing-display">
#                 ⏱️ <strong>Response Time:</strong> Ready to process your first request
#             </div>
#             """)
        
#         query_metadata = gr.Markdown("", elem_classes=["stats-card"])
    
#     # Chat History Section
#     with gr.Group(elem_classes=["section-card"]):
#         gr.HTML('<h2 class="section-header">💭 Conversation History</h2>')
        
#         chat_interface = gr.Chatbot(
#             label="🤖 AI Assistant",
#             height=500,
#             show_label=True,
#             show_share_button=True,
#             show_copy_button=True,
#             avatar_images=["https://cdn-icons-png.flaticon.com/512/1077/1077114.png", "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"]
#         )
    
#     # Sources & Citations Section
#     with gr.Group(elem_classes=["section-card"]):
#         gr.HTML('<h2 class="section-header">📖 Sources & Citations</h2>')
        
#         sources_display = gr.Markdown(
#             "💡 Sources will appear here after you ask questions...",
#             elem_classes=["stats-card"]
#         )
    
#     # Event Handlers with enhanced functionality
    
#     # Real-time text statistics
#     text_input.change(
#         fn=rag_interface.get_detailed_stats,
#         inputs=[text_input],
#         outputs=[text_stats_display]
#     )
    
#     # API health check
#     health_check_btn.click(
#         fn=lambda: rag_interface.check_api_health(),
#         outputs=[api_health_status, has_knowledge_base]
#     )
    
#     # Add text to knowledge base
#     add_text_btn.click(
#         fn=rag_interface.add_text_with_progress,
#         inputs=[text_input, has_knowledge_base],
#         outputs=[kb_status_display, text_input, has_knowledge_base, uploaded_content_display, timing_display, question_status]
#     ).then(
#         fn=lambda has_kb: "➕ Add More Content" if has_kb else "🚀 Create Knowledge Base",
#         inputs=[has_knowledge_base],
#         outputs=[add_text_btn]
#     )
    
#     # Ask question
#     ask_btn.click(
#         fn=rag_interface.query_with_enhanced_response,
#         inputs=[question_input, has_knowledge_base, chat_interface],
#         outputs=[chat_interface, question_input, question_status, query_metadata, timing_display, sources_display]
#     )
    
#     # Enter key for questions
#     question_input.submit(
#         fn=rag_interface.query_with_enhanced_response,
#         inputs=[question_input, has_knowledge_base, chat_interface],
#         outputs=[chat_interface, question_input, question_status, query_metadata, timing_display, sources_display]
#     )
    
#     # Clear chat
#     clear_chat_btn.click(
#         fn=lambda: ([], "", "", "", ""),
#         outputs=[chat_interface, question_status, query_metadata, sources_display, timing_display]
#     )
    
#     # Clear all data
#     clear_all_btn.click(
#         fn=rag_interface.clear_all_data,
#         outputs=[kb_status_display, text_input, has_knowledge_base, uploaded_content_display, timing_display]
#     ).then(
#         fn=lambda: "🚀 Create Knowledge Base",
#         outputs=[add_text_btn]
#     )
    
#     # Initial load
#     demo.load(
#         fn=lambda: rag_interface.check_api_health(),
#         outputs=[api_health_status, has_knowledge_base]
#     )

# # Launch configuration
# if __name__ == "__main__":
#     print("🚀 Launching Advanced RAG Interface...")
#     print(f"📡 API Endpoint: {API_BASE_URL}")
#     print("🌐 Interface URL: http://localhost:7860")
#     print("✨ Enhanced features: Real-time stats, performance monitoring, rich UI")
    
#     demo.launch(
#         server_name="0.0.0.0",
#         server_port=7860,
#         share=False,
#         debug=True,
#         show_error=True,
#         inbrowser=True,
#         favicon_path=None,
#         app_kwargs={"title": "Advanced RAG Chat System"}
#     )
