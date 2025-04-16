from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
from dotenv import load_dotenv

from chroma_store import create_ensemble_retriever, update_retriever_weights

load_dotenv()

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(texts)} chunks")
    return texts

def create_vector_store(texts):
    retriever = create_ensemble_retriever(texts)
    return retriever

def load_existing_index(embeddings=None):
    try:
        retriever = create_ensemble_retriever()
        return retriever
    except Exception as e:
        print(f"Error loading existing index: {str(e)}")
        return None

def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Load and process the PDF
    documents = load_pdf(tmp_file_path)

    # Split text into chunks
    texts = split_documents(documents)

    # Create ensemble retriever
    retriever = create_vector_store(texts)

    return retriever, tmp_file_path
