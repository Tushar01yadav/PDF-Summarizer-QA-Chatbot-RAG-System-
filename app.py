import streamlit as st
import os
import tempfile
import json
from datetime import datetime
import time
import pandas as pd

# Import your RAG system (make sure the original file is named pdf_rag_system.py)
from pdf_rag_system import PDFRAGSystem

# Page configuration
st.set_page_config(
    page_title="PDF RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        padding: 0.5rem;
        border-left: 4px solid #667eea;
        background-color: #323640;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
    
    .stButton > button {
        background-color: #667eea;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #5a6fd8;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'api_key_set' not in st.session_state:
        st.session_state.api_key_set = False

def setup_rag_system(api_key):
    """Initialize the RAG system with API key"""
    try:
        st.session_state.rag_system = PDFRAGSystem(mistral_api_key=api_key)
        st.session_state.api_key_set = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return False

def process_uploaded_pdf(uploaded_file, document_name):
    """Process an uploaded PDF file"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Process the PDF
        with st.spinner(f'Processing "{document_name}"... This may take a few minutes.'):
            result = st.session_state.rag_system.process_pdf(temp_file_path, document_name)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Add to processed documents
        st.session_state.processed_documents.append({
            'name': document_name,
            'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'pages': result['total_pages'],
            'content_items': result['total_content_items'],
            'breakdown': result['content_breakdown']
        })
        
        return result
    
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise e

def display_document_stats(doc_info):
    """Display document processing statistics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pages", doc_info['pages'])
    
    with col2:
        st.metric("Content Items", doc_info['content_items'])
    
    with col3:
        st.metric("Text Sections", doc_info['breakdown']['text_items'])
    
    with col4:
        st.metric("Images", 
                 doc_info['breakdown']['embedded_images'] + 
                 doc_info['breakdown']['full_page_images'])

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📚 PDF RAG System</h1>', unsafe_allow_html=True)
    st.markdown("**Upload PDFs, process them with AI, and ask intelligent questions!**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<div class="section-header">🔧 Configuration</div>', unsafe_allow_html=True)
        
        # API Key input
        api_key = st.text_input(
            "Mistral API Key", 
            type="password",
            help="Enter your Mistral API key to use the vision and language models"
        )
        
        if api_key and not st.session_state.api_key_set:
            if setup_rag_system(api_key):
                st.success("✅ API key configured successfully!")
            else:
                st.error("❌ Failed to configure API key")
        
        # Document management section
        st.markdown('<div class="section-header">📄 Processed Documents</div>', unsafe_allow_html=True)
        
        if st.session_state.processed_documents:
            for i, doc in enumerate(st.session_state.processed_documents):
                with st.expander(f"📁 {doc['name']}"):
                    st.write(f"**Processed:** {doc['processed_at']}")
                    st.write(f"**Pages:** {doc['pages']}")
                    st.write(f"**Content Items:** {doc['content_items']}")
                    
                    # Breakdown
                    breakdown = doc['breakdown']
                    st.write("**Content Breakdown:**")
                    st.write(f"- Text sections: {breakdown['text_items']}")
                    st.write(f"- Embedded images: {breakdown['embedded_images']}")
                    st.write(f"- Full page images: {breakdown['full_page_images']}")
        else:
            st.info("No documents processed yet")
        
        # Clear all data
        if st.button("🗑️ Clear All Data"):
            st.session_state.processed_documents = []
            st.session_state.chat_history = []
            if st.session_state.rag_system:
                st.session_state.rag_system.documents_db = []
            st.rerun()
    
    # Main content area
    if not st.session_state.api_key_set:
        st.markdown('<div class="info-box">Please enter your Mistral API key in the sidebar to get started.</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("### How to get a Mistral API Key:")
        st.markdown("""
        1. Visit [console.mistral.ai](https://console.mistral.ai)
        2. Create an account or sign in
        3. Navigate to API Keys section
        4. Create a new API key
        5. Copy and paste it in the sidebar
        """)
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "💬 Chat with PDFs", "📊 Analytics"])
    
    # Tab 1: Upload and Process
    with tab1:
        st.markdown('<div class="section-header">📤 Upload PDF Documents</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to process with the RAG system"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    default_name = uploaded_file.name.replace('.pdf', '')
                    document_name = st.text_input(
                        f"Document name for {uploaded_file.name}:",
                        value=default_name,
                        key=f"name_{uploaded_file.name}"
                    )
                
                with col2:
                    process_btn = st.button(
                        "🔄 Process",
                        key=f"process_{uploaded_file.name}",
                        help="Process this PDF with AI analysis"
                    )
                
                if process_btn and document_name:
                    try:
                        result = process_uploaded_pdf(uploaded_file, document_name)
                        
                        # Display success message
                        st.markdown('<div class="success-box">✅ PDF processed successfully!</div>', 
                                   unsafe_allow_html=True)
                        
                        # Display statistics
                        display_document_stats({
                            'pages': result['total_pages'],
                            'content_items': result['total_content_items'],
                            'breakdown': result['content_breakdown']
                        })
                        
                        # Save database
                        st.session_state.rag_system.save_all_documents()
                        
                        st.success(f"Document '{document_name}' is ready for questions!")
                        
                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ Error processing PDF: {str(e)}</div>', 
                                   unsafe_allow_html=True)
    
    # Tab 2: Chat Interface
    with tab2:
        st.markdown('<div class="section-header">💬 Ask Questions</div>', unsafe_allow_html=True)
        
        if not st.session_state.processed_documents:
            st.info("Please process at least one PDF document first to start asking questions.")
        else:
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("### 💭 Conversation History")
                for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
                    with st.expander(f"Q{i+1}: {question[:50]}{'...' if len(question) > 50 else ''}"):
                        st.markdown(f"**Question:** {question}")
                        st.markdown(f"**Answer:** {answer}")
                        if sources:
                            st.markdown(f"**Sources:** {', '.join(sources)}")
            
            # Question input
            st.markdown("### ❓ Ask a Question")
            question = st.text_area(
                "Enter your question about the processed documents:",
                height=100,
                placeholder="e.g., What is the main topic of the document? What are the key findings?"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_button = st.button("🤔 Ask Question", type="primary")
            with col2:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            if ask_button and question.strip():
                try:
                    with st.spinner("🔍 Searching for relevant information and generating answer..."):
                        # Get relevant pages
                        relevant_pages = st.session_state.rag_system.query_documents(question, top_k=3)
                        
                        if not relevant_pages:
                            st.warning("❌ No relevant information found in the processed documents.")
                        else:
                            # Show relevant content
                            with st.expander("📋 Relevant Content Found"):
                                for i, item in enumerate(relevant_pages, 1):
                                    st.write(f"**{i}. Page {item['page_number']} ({item['content_type']}) - Similarity: {item['similarity']:.3f}**")
                                    st.write(f"From: {item['document_name']}")
                                    st.write(f"Preview: {item['caption'][:200]}...")
                                    st.divider()
                            
                            # Generate answer
                            answer = st.session_state.rag_system.answer_question(question, relevant_pages)
                            
                            # Display answer
                            st.markdown("### 💡 Answer")
                            st.markdown(answer)
                            
                            # Show sources
                            sources = [f"Page {p['page_number']} ({p['content_type']})" for p in relevant_pages]
                            st.markdown(f"**📚 Sources:** {', '.join(sources)}")
                            
                            # Add to chat history
                            st.session_state.chat_history.append((question, answer, sources))
                
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
    
    # Tab 3: Analytics
    with tab3:
        st.markdown('<div class="section-header">📊 Document Analytics</div>', unsafe_allow_html=True)
        
        if not st.session_state.processed_documents:
            st.info("No documents processed yet.")
        else:
            # Create summary statistics
            total_docs = len(st.session_state.processed_documents)
            total_pages = sum(doc['pages'] for doc in st.session_state.processed_documents)
            total_content = sum(doc['content_items'] for doc in st.session_state.processed_documents)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", total_docs)
            with col2:
                st.metric("Total Pages", total_pages)
            with col3:
                st.metric("Total Content Items", total_content)
            
            # Document breakdown chart
            if total_docs > 0:
                st.subheader("📈 Content Type Distribution")
                
                # Aggregate content types
                text_items = sum(doc['breakdown']['text_items'] for doc in st.session_state.processed_documents)
                embedded_images = sum(doc['breakdown']['embedded_images'] for doc in st.session_state.processed_documents)
                full_page_images = sum(doc['breakdown']['full_page_images'] for doc in st.session_state.processed_documents)
                
                # Create DataFrame for chart
                chart_data = pd.DataFrame({
                    'Content Type': ['Text Sections', 'Embedded Images', 'Full Page Images'],
                    'Count': [text_items, embedded_images, full_page_images]
                })
                
                st.bar_chart(chart_data.set_index('Content Type'))
                
                # Document details table
                st.subheader("📋 Document Details")
                df = pd.DataFrame(st.session_state.processed_documents)
                st.dataframe(df, use_container_width=True)
                
                # Chat analytics
                if st.session_state.chat_history:
                    st.subheader("💬 Chat Statistics")
                    st.metric("Total Questions Asked", len(st.session_state.chat_history))
                    
                    # Most recent questions
                    st.write("**Recent Questions:**")
                    for question, _, _ in st.session_state.chat_history[-5:]:
                        st.write(f"• {question}")

if __name__ == "__main__":
    main()