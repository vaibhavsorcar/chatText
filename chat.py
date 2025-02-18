import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import time
import base64

# Loading environment variables
load_dotenv()

def load_image(image_path):
    """Load an image and convert it to base64 format"""
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded_image}"

# Constants for file size limit and others
MAX_FILE_SIZE_MB = 50  

def display_header():
    """Displays the header of the app with custom round image as the icon"""
    icon_path = "icon.jpg"  # Path to your icon.jpg
    icon_base64 = load_image(icon_path)
    
    # Set the icon for the app in the browser tab (this will not be round)
    st.set_page_config(page_title="Chat with PDF", page_icon=icon_base64, layout="wide")
    
    # Custom styling to place the round icon on top of the page
    st.markdown(
        """
        <style>
        .round-icon {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            border: 2px solid #fff;
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-top: 20px;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Display the round icon at the top center
    st.markdown(f'<img src="{icon_base64}" class="round-icon">', unsafe_allow_html=True)
    
    # Display the title below the icon
    st.markdown('<h1 class="title">Chat with Your PDF 📚</h1>', unsafe_allow_html=True)

def display_sidebar():
    """Displays the sidebar with user options"""
    st.sidebar.title("Options")
    st.sidebar.markdown("### Upload Your PDF Files Here:")
    st.sidebar.markdown("""
        - Upload multiple PDF files for processing.
        - Click **Submit & Process** once you've uploaded the files.
        - Ask questions related to your PDF files below.
    """)
    st.sidebar.markdown("Made with ❤️ by Chat for PDFs")

def display_file_uploader():
    """Handle the PDF file uploader with file size validation"""
    uploaded_files = st.file_uploader(
        "Upload PDF Files (up to 200MB)", type="pdf", accept_multiple_files=True
    )
    
    if uploaded_files:
        # Check file sizes
        for file in uploaded_files:
            if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"{file.name} is too large. Max size is {MAX_FILE_SIZE_MB}MB.")
                return []
        return uploaded_files
    return []

def get_pdf_text(pdf_docs):
    """Extract text from PDFs"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split the extracted text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and store vector database"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Return the conversational chain for the QA system"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    """Handle user input and provide the answer"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)
    
    return response["output_text"]

def main():
    display_header()
    display_sidebar()

    # Handle file uploading
    uploaded_files = display_file_uploader()
    if uploaded_files:
        st.info("Processing your PDF files... Please wait.")
        
        with st.spinner("Processing PDFs..."):
            progress_bar = st.progress(0)
            raw_text = get_pdf_text(uploaded_files)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("PDF Processing completed successfully!")
        
        # Question Section
        st.subheader("Ask a Question about your PDF")
        user_question = st.text_input("Enter your question:")
        
        if user_question:
            st.info("Fetching the answer...")
            with st.spinner("Generating the answer..."):
                answer = user_input(user_question)
                st.write(f"**Answer**: {answer}")

    else:
        st.info("Please upload PDF files to begin.")

if __name__ == "__main__":
    main()
