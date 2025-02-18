import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Load environment variables
load_dotenv()

def create_custom_theme():
    """Create dark theme with modern UI elements"""
    return """
    <style>
        /* Dark Theme */
        .stApp {
            background: #1E1E1E;
            color: #E0E0E0;
        }
        
        /* Main Container */
        .main-container {
            background: #2D2D2D;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #4A90E2 !important;
            font-weight: 600 !important;
        }
        
        /* Custom Button */
        .custom-button {
            background: linear-gradient(45deg, #4A90E2, #45B7D1);
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .custom-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
        }
        
        /* File Uploader */
        .uploadedFile {
            background: #383838 !important;
            border: 1px solid #4A4A4A !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }
        
        /* Chat Area */
        .chat-message {
            background: #383838;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #4A90E2;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background: #2D2D2D;
        }
        
        /* Input Fields */
        .stTextInput input {
            background: #383838 !important;
            color: #E0E0E0 !important;
            border: 1px solid #4A4A4A !important;
            border-radius: 6px !important;
        }
        
        /* Progress Bar */
        .stProgress > div > div {
            background-color: #4A90E2 !important;
        }
        
        /* Success Message */
        .success-message {
            background: #2D4A3E;
            color: #A7F3D0;
            padding: 10px;
            border-radius: 6px;
            margin: 10px 0;
        }
        
        /* Error Message */
        .error-message {
            background: #4A2D2D;
            color: #F3A7A7;
            padding: 10px;
            border-radius: 6px;
            margin: 10px 0;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .fade-in {
            animation: fadeIn 0.3s ease-out;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main-container {
                padding: 10px;
                margin: 5px 0;
            }
            
            .chat-message {
                padding: 10px;
            }
        }
    </style>
    """

class PDFProcessor:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    def process_pdf(self, pdf_file):
        try:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error processing {pdf_file.name}: {str(e)}")
            return ""

    def process_pdfs_parallel(self, pdf_files):
        with ThreadPoolExecutor() as executor:
            texts = list(executor.map(self.process_pdf, pdf_files))
        return " ".join(texts)

def create_conversational_chain():
    prompt_template = """
    Please provide a detailed answer based on the context provided. If the information is not in the context, clearly state that.
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def main():
    # Apply custom theme
    st.markdown(create_custom_theme(), unsafe_allow_html=True)
    
    # App Header
    st.markdown("""
        <div class="main-container fade-in">
            <h1>💡 Smart PDF Chat</h1>
            <p style="color: #B0B0B0;">Upload your PDFs and start asking questions instantly.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize PDF processor
    pdf_processor = PDFProcessor()
    
    # Sidebar with dark theme
    # with st.sidebar:
    #     st.markdown('<div style="background: #2D2D2D; padding: 20px; border-radius: 10px;">', unsafe_allow_html=True)
    #     st.title("⚙️ Settings")
        
    #     # Theme toggle
    #     theme_mode = st.select_slider(
    #         "Appearance",
    #         options=["Darker", "Dark", "Light"],
    #         value="Dark"
    #     )
        
    #     # Performance settings
    #     st.subheader("Performance")
    #     use_cache = st.toggle("Enable Caching", value=True)
        
    #     st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content area
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # File upload with progress
    uploaded_files = st.file_uploader(
        "Drop your PDF files here",
        type="pdf",
        accept_multiple_files=True,
        help="📁 Supports multiple PDF files"
    )
    
    if uploaded_files:
        with st.spinner("📄 Processing documents..."):
            # Process PDFs with progress bar
            progress_bar = st.progress(0)
            
            # Generate hash for caching
            files_hash = hashlib.md5(b"".join(file.read() for file in uploaded_files)).hexdigest()
            
            # Reset file pointers
            for file in uploaded_files:
                file.seek(0)
            
            # Process text
            text = pdf_processor.process_pdfs_parallel(uploaded_files)
            progress_bar.progress(50)
            
            # Create text chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            text_chunks = text_splitter.split_text(text)
            
            # Create vector store
            vector_store = FAISS.from_texts(text_chunks, embedding=pdf_processor.embeddings)
            progress_bar.progress(100)
            
            st.markdown("""
                <div class="success-message fade-in">
                    ✅ Documents processed successfully!
                </div>
            """, unsafe_allow_html=True)
        
        # Chat interface
        st.markdown('<div class="chat-container fade-in">', unsafe_allow_html=True)
        st.subheader("🤖 Ask your questions")
        
        user_question = st.text_input(
            "",
            placeholder="Type your question here...",
            key="question_input"
        )
        
        if user_question:
            with st.spinner("🤔 Thinking..."):
                # Get response
                docs = vector_store.similarity_search(user_question)
                chain = create_conversational_chain()
                response = chain(
                    {"input_documents": docs, "question": user_question},
                    return_only_outputs=True
                )
                
                # Display response
                st.markdown(f"""
                    <div class="chat-message fade-in">
                        <strong>Answer:</strong><br>
                        {response["output_text"]}
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()