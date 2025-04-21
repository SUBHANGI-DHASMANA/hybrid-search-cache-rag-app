import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
import os
import time
import shutil
from dotenv import load_dotenv

load_dotenv()

from data_ingestion import process_uploaded_file, load_existing_index

st.title("Research Paper Q&A Assistant")

# Initialize session state variables if they don't exist
if 'response_cache' not in st.session_state:
    st.session_state.response_cache = {}

# Track if an answer has been displayed
if 'answer_displayed' not in st.session_state:
    st.session_state.answer_displayed = False

# Store the last question for context
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""

# Track if we're currently processing a question
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Add buttons for clearing data
col1, col2 = st.columns(2)

# Button to clear ChromaDB data
with col1:
    if st.button("Clear Research Paper Database"):
        chroma_dir = os.path.join(os.getcwd(), "chroma_db")
        if os.path.exists(chroma_dir):
            try:
                shutil.rmtree(chroma_dir)
                st.success("Research paper database cleared successfully!")
                # Reset the retriever
                retriever = None
            except Exception as e:
                st.error(f"Error clearing ChromaDB data: {str(e)}")
        else:
            st.info("No ChromaDB data to clear.")

    if st.button("Clear Analysis Cache"):
        try:
            st.session_state.response_cache = {}
            # Reset the Q&A session state
            st.session_state.answer_displayed = False
            st.session_state.last_question = ""
            st.success("Analysis cache cleared successfully!")
        except Exception as e:
            st.error(f"Error clearing cache: {str(e)}")


# Initialize models
@st.cache_resource
def initialize_models():
    llm = Ollama(model="mistral", temperature=0.2, num_ctx=2048)  # Lower temperature for more factual responses
    embeddings = OllamaEmbeddings(model="all-minilm")
    return llm, embeddings

llm, embeddings = initialize_models()

st.markdown("### Upload a research paper to analyze and ask questions about its content.")
st.markdown("This tool helps you extract insights, understand methodologies, and explore findings from academic papers.")

pdf_file = st.file_uploader("Upload your research paper (PDF format)", type=['pdf'])

retriever = None
tmp_file_path = None

try:
    with st.spinner("Checking for previously analyzed research papers..."):
        existing_retriever = load_existing_index()
        if existing_retriever:
            retriever = existing_retriever
            st.success("Found previously analyzed research papers!")
except Exception as e:
    st.error(f"Error checking for previous research papers: {str(e)}")

if pdf_file is not None:
    with st.spinner("Processing research paper and extracting content for analysis..."):
        try:
            retriever, tmp_file_path = process_uploaded_file(pdf_file)
            st.success("Research paper processed and ready for analysis!")
            # Reset the session state for a new paper
            st.session_state.answer_displayed = False
            st.session_state.last_question = ""
        except Exception as e:
            st.error(f"Error processing research paper: {str(e)}")

if retriever is not None:
    bm25_weight = 0.5
    vector_weight = 0.5
    from chroma_store import update_retriever_weights
    retriever = update_retriever_weights(retriever, bm25_weight, vector_weight)
    # Always show the Research Paper Analysis header
    st.markdown("### Research Paper Analysis")

    # Create columns for the question input and Ask button
    question_col, button_col = st.columns([4, 1])

    with question_col:
        question = st.text_input("Ask a question about the research paper (e.g., 'What methodology was used?', 'What are the key findings?', 'What limitations were discussed?'):")

    with button_col:
        # Only show the button if we're not already processing a question
        ask_button = st.button("Ask", key="ask_button", use_container_width=True, disabled=st.session_state.get('processing', False))

    if 'response_cache' not in st.session_state:
        st.session_state.response_cache = {}

    # Process the question when the Ask button is clicked or when a question is entered
    if (question and ask_button) or (question and not st.session_state.answer_displayed):
        try:
            # Set processing flag to disable the button during processing
            st.session_state.processing = True
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

                # Research paper specific prompt
                prompt = f"""You are a research assistant analyzing academic papers. Use the following excerpts from a research paper to answer the question.

Paper excerpts:
{context}

Question: {question}

Provide a comprehensive answer based on the paper content. Include relevant details such as:
- Key findings or arguments related to the question
- Methodologies or approaches mentioned
- Data or evidence presented
- Limitations or future work discussed

If the information is not available in the provided excerpts, state that clearly. Do not make up information not present in the paper.

Format your answer in a clear, academic style with proper citations to sections of the paper when relevant.

Answer:"""

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

            # Display the answer in a more academic format
            st.write("### Research Analysis:")
            st.markdown(answer)

            # Add citation information
            st.write("### Source Information:")
            st.info("The analysis above is based on the uploaded research paper and the specific sections referenced in the 'Paper Excerpts' below.")

            # Display the source documents with better formatting
            with st.expander("View Paper Excerpts"):
                for i, doc in enumerate(docs[:4]):
                    # Extract page number if available in metadata
                    page_num = doc.metadata.get('page', i+1) if hasattr(doc, 'metadata') and doc.metadata else i+1
                    st.markdown(f"**Excerpt {i+1} (Page {page_num})**")

                    # Format the text better
                    st.markdown("```")
                    st.markdown(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    st.markdown("```")
                    st.markdown("---")

            # Set the answer displayed flag and store the last question
            st.session_state.answer_displayed = True
            st.session_state.last_question = question
            # Reset processing flag
            st.session_state.processing = False
        except Exception as e:
            # Reset processing flag on error
            st.session_state.processing = False
            st.error(f"An error occurred: {str(e)}")

    if tmp_file_path:
        os.unlink(tmp_file_path)