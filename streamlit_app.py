import streamlit as st
import os
import glob
import time # For basic progress simulation

# --- Core Processing Logic Imports (Keep these) ---
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # Use community for HuggingFace
from langchain_community.vectorstores import FAISS # Use community for FAISS
# Use updated imports if using recent langchain-openai install
from langchain_openai import ChatOpenAI # Updated import for OpenAI
# from langchain_community.llms import HuggingFaceHub # Use community for HuggingFace Hub if needed
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# --- Configuration & Constants ---
load_dotenv() # Load .env file if present (mainly for local testing)
DOCUMENTS_DIR = "uploaded_docs" # Directory to save uploaded files temporarily
INDEX_PATH = "./faiss_index_streamlit" # Use a separate index path for the Streamlit app
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Ensure the directories exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(INDEX_PATH, exist_ok=True)

# --- Tesseract Check (Optional but Recommended) ---
# Add your Tesseract check code here if needed, using st.error() to display messages
# try:
#     tesseract_version = pytesseract.get_tesseract_version()
# except Exception as e:
#     st.error(f"Tesseract OCR Error: {e}. Ensure Tesseract is installed and accessible.")
#     st.stop() # Stop execution if Tesseract isn't working

# --- Caching Expensive Operations ---
# Cache the embeddings model loading
@st.cache_resource # Caches the resource across reruns
def load_embeddings(model_name):
    st.info(f"Loading embedding model: {model_name}...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        st.success("Embedding model loaded.")
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        st.stop()

# --- Helper Functions (Keep these, adjust slightly for Streamlit feedback) ---
def ocr_pdf_to_text(pdf_path):
    """ Performs OCR on a PDF file and returns text content with page numbers. """
    basename = os.path.basename(pdf_path)
    st.write(f"  Performing OCR on: {basename}...") # Use st.write for output
    extracted_docs = []
    try:
        images = convert_from_path(pdf_path, dpi=300)
        progress_bar = st.progress(0, text=f"OCR Progress for {basename}")
        for i, image in enumerate(images):
            try:
                text = pytesseract.image_to_string(image, lang='eng')
                if text.strip():
                    metadata = {"source": basename, "page": i + 1}
                    extracted_docs.append(Document(page_content=text, metadata=metadata))
                progress_bar.progress((i + 1) / len(images), text=f"OCR Progress for {basename}: Page {i+1}/{len(images)}")
            except Exception as e:
                 st.warning(f"    Error OCR'ing page {i+1} of {basename}: {e}")
        progress_bar.empty() # Remove progress bar when done
        st.write(f"    Finished OCR. Extracted text from {len(extracted_docs)} pages.")
        return extracted_docs
    except Exception as e:
        st.error(f"  Error converting PDF to images for {basename} (check Poppler install): {e}")
        if 'progress_bar' in locals(): progress_bar.empty()
        return []

def load_and_process_single_doc(file_path):
    """Loads, OCRs (if needed), and returns documents from a single file."""
    docs = []
    file_name = os.path.basename(file_path)
    st.write(f"Processing document: {file_name}")

    if file_path.lower().endswith(".pdf"):
        try:
            standard_loader = PyPDFLoader(file_path)
            pdf_docs = standard_loader.load()
            is_likely_scanned = False
            if not pdf_docs:
                 is_likely_scanned = True
                 st.write(f"  No text found via standard parser. Attempting OCR.")
            else:
                total_chars = sum(len(doc.page_content) for doc in pdf_docs)
                avg_chars_per_page = total_chars / len(pdf_docs) if pdf_docs else 0
                if avg_chars_per_page < 100:
                    st.write(f"  Low text content ({avg_chars_per_page:.0f} avg chars/page). Assuming scanned, attempting OCR.")
                    is_likely_scanned = True

            if is_likely_scanned:
                ocr_docs = ocr_pdf_to_text(file_path)
                if ocr_docs: docs.extend(ocr_docs)
                else: st.warning(f"  OCR failed or yielded no text for {file_name}.")
            else:
                st.write(f"  Loaded {len(pdf_docs)} pages using standard PDF parser.")
                docs.extend(pdf_docs)
        except Exception as e:
            st.error(f"  Error with standard PDF processing for {file_name}: {e}. Trying OCR fallback.")
            ocr_docs = ocr_pdf_to_text(file_path)
            if ocr_docs: docs.extend(ocr_docs)
            else: st.warning(f"  Fallback OCR also failed for {file_name}.")

    elif file_path.lower().endswith(".txt"):
        try:
            loader = TextLoader(file_path)
            docs.extend(loader.load())
            st.write(f"  Loaded content from TXT file.")
        except Exception as e:
            st.error(f"  Error loading TXT file {file_name}: {e}")
    else:
        st.warning(f"  Skipping unsupported file type: {file_name}")

    return docs

# --- Custom Prompt Template (Keep or modify as needed) ---
custom_prompt_template = """You are a helpful and insightful AI assistant...

    Context:
    {context}

    Question: {question}

    Insightful Answer:""" # Keep your preferred prompt text

PROMPT = PromptTemplate(
    template=custom_prompt_template, input_variables=["context", "question"]
)


# ==============================================================================
# Streamlit Application UI and Logic
# ==============================================================================

st.set_page_config(page_title="Document Q&A Assistant", layout="wide")
st.title("📄 Document Q&A and Strategy Assistant")
st.write("Upload your documents (PDF, TXT), and ask questions or request analysis based on their content.")

# --- Initialize session state variables ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "known_sources" not in st.session_state:
    st.session_state.known_sources = set()
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# --- Load Embeddings ---
embeddings = load_embeddings(EMBEDDING_MODEL_NAME)

# --- Sidebar for Configuration and Upload ---
with st.sidebar:
    st.header("Configuration")

    # Use Streamlit secrets for API keys in deployed apps
    # For local testing, it falls back to .env or environment variables
    openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    # hf_api_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN"))

    if not openai_api_key:
         st.warning("OpenAI API Key not found. Please add it to Streamlit secrets or your environment variables.")
         # Add similar check for HF token if using HuggingFaceHub LLM

    # LLM Temperature Slider
    llm_temperature = st.slider(
        "LLM Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.4, # Default temperature from previous code
        step=0.1,
        help="Lower values are more factual/deterministic, higher values are more creative/random."
    )

    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload new documents here. Previously processed documents will be remembered for this session."
    )

    process_button = st.button("Process Uploaded Documents")

# --- Document Processing Logic ---
if process_button and uploaded_files:
    new_files_processed = False
    with st.spinner("Processing documents... This may take a while for OCR."):
        # Save uploaded files temporarily
        saved_file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_file_paths.append(file_path)

        # Identify files that haven't been processed yet in this session/index
        new_files_to_process = []
        current_known_sources = st.session_state.get("known_sources", set()) # Get current state
        for file_path in saved_file_paths:
             file_name = os.path.basename(file_path)
             # Check if not already processed (based on filename)
             if file_name not in current_known_sources:
                 new_files_to_process.append(file_path)

        new_docs = []
        if new_files_to_process:
            st.write(f"Found {len(new_files_to_process)} new documents to process:")
            for file_path in new_files_to_process:
                processed_docs = load_and_process_single_doc(file_path)
                if processed_docs:
                    new_docs.extend(processed_docs)
                    st.session_state.known_sources.add(os.path.basename(file_path)) # Add to known sources
                    new_files_processed = True # Flag that we processed something
        else:
            st.info("No *new* documents detected among uploads.")

        # Split new documents if any were processed
        new_texts = []
        if new_docs:
            st.write("Splitting new documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            new_texts = text_splitter.split_documents(new_docs)
            st.write(f"Split into {len(new_texts)} chunks.")

        # Update or Create Vector Store only if new texts exist
        if new_texts:
            st.write("Updating vector store...")
            if st.session_state.vector_store is None:
                # Attempt to load first, in case it exists but wasn't in session state
                if os.path.exists(INDEX_PATH) and os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
                    try:
                        st.session_state.vector_store = FAISS.load_local(
                            INDEX_PATH, embeddings, allow_dangerous_deserialization=True
                        )
                        st.write("Loaded existing index from disk.")
                        # Refresh known sources from loaded index (optional but good practice)
                        if hasattr(st.session_state.vector_store, 'docstore'):
                             st.session_state.known_sources.update(
                                 {d.metadata.get('source') for d in st.session_state.vector_store.docstore._dict.values() if d.metadata.get('source')}
                             )

                    except Exception as e:
                        st.warning(f"Could not load index from disk ({e}), will create a new one.")
                        st.session_state.vector_store = None

            # Add to existing or create new
            try:
                if st.session_state.vector_store is not None:
                    st.session_state.vector_store.add_documents(new_texts)
                    st.write(f"Added {len(new_texts)} new chunks to the vector store.")
                else:
                    st.session_state.vector_store = FAISS.from_documents(new_texts, embeddings)
                    st.write("Created a new vector store.")

                # Save the updated/new index
                st.session_state.vector_store.save_local(INDEX_PATH)
                st.success("Vector store updated and saved.")
            except Exception as e:
                st.error(f"Error updating/saving vector store: {e}")
        elif not new_files_processed:
             st.write("No new documents were processed.")
        else: # new_files_processed is True, but no new_texts generated (e.g., empty files)
             st.warning("Processed new files but generated no content chunks.")


        # --- Initialize/Update LLM and QA Chain ---
        # Re-initialize LLM if temperature changed or it doesn't exist
        # A more robust approach might compare current temp to stored temp
        if st.session_state.llm is None or st.session_state.llm.temperature != llm_temperature:
            st.write("Initializing LLM...")
            try:
                # Using OpenAI - adapt if using HuggingFaceHub
                if openai_api_key:
                    st.session_state.llm = ChatOpenAI(
                        model_name="gpt-3.5-turbo", # Consider gpt-4
                        temperature=llm_temperature,
                        openai_api_key=openai_api_key
                    )
                    st.write(f"Using OpenAI model: {st.session_state.llm.model_name} (Temp: {llm_temperature})")
                else:
                     st.error("Cannot initialize LLM: OpenAI API Key missing.")
                     st.session_state.llm = None

            except Exception as e:
                st.error(f"LLM Initialization failed: {e}")
                st.session_state.llm = None

        # Create/update QA chain if vector store and LLM are ready
        if st.session_state.vector_store is not None and st.session_state.llm is not None:
             st.write("Setting up QA chain...")
             retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
             st.session_state.qa_chain = RetrievalQA.from_chain_type(
                 llm=st.session_state.llm,
                 chain_type="stuff",
                 retriever=retriever,
                 return_source_documents=True,
                 chain_type_kwargs={"prompt": PROMPT}
             )
             st.success("Assistant is ready to answer questions!")
        else:
             st.warning("QA chain not ready. Ensure documents are processed and LLM is initialized.")
             st.session_state.qa_chain = None


# --- Q&A Interface ---
st.header("Ask Questions or Assign Tasks")

query = st.text_input("Enter your question or task:", key="query_input")

if query:
    if st.session_state.qa_chain:
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.qa_chain({"query": query})
                st.subheader("Response:")
                st.markdown(result.get("result", "No answer generated.")) # Use markdown for better formatting

                st.subheader("Retrieved Context Sources:")
                source_docs = result.get("source_documents")
                if source_docs:
                    with st.expander("Show Sources"): # Use expander to hide details initially
                        unique_sources = {}
                        for doc in source_docs:
                            source = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "N/A")
                            source_key = f"{source} (Page: {page})"
                            if source_key not in unique_sources:
                                snippet = doc.page_content[:200].replace('\n', ' ').strip() + "..."
                                unique_sources[source_key] = snippet
                        if not unique_sources: st.write(" - No specific source documents retrieved.")
                        else:
                             for source_key, snippet in unique_sources.items():
                                 st.write(f"**{source_key}**")
                                 st.caption(f"Snippet: {snippet}")
                else:
                    st.write(" - Source documents not returned by the chain.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please process some documents first using the sidebar.")

# --- Display Known Sources ---
st.sidebar.header("Processed Documents")
if st.session_state.known_sources:
    for src in sorted(list(st.session_state.known_sources)):
        st.sidebar.write(f"- {src}")
else:
    st.sidebar.info("No documents processed in this session yet.")
