import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
import os
import time
import shutil
from dotenv import load_dotenv

load_dotenv()

from data_ingestion import process_uploaded_file, load_existing_index

st.title("Hybrid Search Cache RAG Application")

# Initialize session state for response cache if it doesn't exist
if 'response_cache' not in st.session_state:
    st.session_state.response_cache = {}

# Add buttons for clearing data
col1, col2 = st.columns(2)

# Button to clear ChromaDB data
with col1:
    if st.button("Clear ChromaDB Data"):
        chroma_dir = os.path.join(os.getcwd(), "chroma_db")
        if os.path.exists(chroma_dir):
            try:
                shutil.rmtree(chroma_dir)
                st.success("ChromaDB data cleared successfully!")
                # Reset the retriever
                retriever = None
            except Exception as e:
                st.error(f"Error clearing ChromaDB data: {str(e)}")
        else:
            st.info("No ChromaDB data to clear.")
    
    if st.button("Clear Cache"):
        try:
            st.session_state.response_cache = {}
            st.success("Response cache cleared successfully!")
        except Exception as e:
            st.error(f"Error clearing cache: {str(e)}")


# Initialize models
@st.cache_resource
def initialize_models():
    llm = Ollama(model="mistral", temperature=0.5, num_ctx=2048)
    embeddings = OllamaEmbeddings(model="all-minilm")
    return llm, embeddings

llm, embeddings = initialize_models()

pdf_file = st.file_uploader("Upload your PDF", type=['pdf'])

retriever = None
tmp_file_path = None

try:
    with st.spinner("Connecting to ChromaDB..."):
        existing_retriever = load_existing_index()
        if existing_retriever:
            retriever = existing_retriever
            st.success("Connected to existing ChromaDB collection with hybrid search!")
except Exception as e:
    st.error(f"Error connecting to ChromaDB: {str(e)}")

if pdf_file is not None:
    with st.spinner("Processing PDF and storing in ChromaDB with hybrid search..."):
        try:
            retriever, tmp_file_path = process_uploaded_file(pdf_file)
            st.success("PDF processed and stored in ChromaDB with hybrid search!")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")

if retriever is not None:
    bm25_weight = 0.5
    vector_weight = 0.5
    from chroma_store import update_retriever_weights
    retriever = update_retriever_weights(retriever, bm25_weight, vector_weight)
    question = st.text_input("Ask a question about your PDF:")

    if 'response_cache' not in st.session_state:
        st.session_state.response_cache = {}

    if question:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_display = st.empty()

            start_time = time.time()

            # Phase 1: Retrieving documents
            phase1_start = time.time()
            status_text.text("Retrieving relevant documents...")
            progress_bar.progress(10)

            # Get documents from retriever directly for faster processing
            docs = retriever.get_relevant_documents(question)
            phase1_end = time.time()
            phase1_time = phase1_end - phase1_start
            progress_bar.progress(40)

            # Update time display
            time_display.text(f"⏱️ Retrieval time: {phase1_time:.2f} seconds")

            # Phase 2: Generating answer
            phase2_start = time.time()
            status_text.text("Generating answer...")

            # Check if we have a cached response
            cache_key = question.strip().lower()

            if cache_key in st.session_state.response_cache:
                # Use cached response
                result = st.session_state.response_cache[cache_key]
                answer = result['answer']
                docs = result['source_documents']
            else:
                context = "\n\n".join([doc.page_content for doc in docs[:4]])
                prompt = f"Context information is below.\n\n{context}\n\nQuestion: {question}\n\nAnswer the question based on the context provided. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nAnswer:"

                # Generate answer
                answer = llm.invoke(prompt)

                # Cache the response
                st.session_state.response_cache[cache_key] = {
                    'answer': answer,
                    'source_documents': docs
                }

            phase2_end = time.time()
            phase2_time = phase2_end - phase2_start

            # Calculate total time
            total_time = time.time() - start_time

            progress_bar.progress(100)
            status_text.text("Done!")

            # Update time display with detailed breakdown
            time_display.text(f"⏱️ Total time: {total_time:.2f} seconds | Retrieval: {phase1_time:.2f}s | Generation: {phase2_time:.2f}s")

            # Display the answer
            st.write("### Answer:")
            st.write(answer)

            # Display the source documents for debugging
            with st.expander("View Source Documents"):
                for i, doc in enumerate(docs[:4]):
                    st.markdown(f"**Document {i+1}**")
                    st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    st.markdown("---")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    if tmp_file_path:
        os.unlink(tmp_file_path)