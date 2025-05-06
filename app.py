import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import time
import shutil
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv()

os.makedirs("offload", exist_ok=True)

from data_ingestion import process_uploaded_file, load_existing_index

st.title("Research Paper Q&A Assistant")
st.caption("Powered by Mistral 7B Instruct Research Paper model")

if 'response_cache' not in st.session_state:
    st.session_state.response_cache = {}

if 'answer_displayed' not in st.session_state:
    st.session_state.answer_displayed = False

if 'last_question' not in st.session_state:
    st.session_state.last_question = ""

if 'processing' not in st.session_state:
    st.session_state.processing = False

col1, col2 = st.columns(2)

with col1:
    if st.button("Clear Research Paper Database"):
        chroma_dir = os.path.join(os.getcwd(), "chroma_db")
        if os.path.exists(chroma_dir):
            try:
                shutil.rmtree(chroma_dir)
                st.success("Research paper database cleared successfully!")
                retriever = None
            except Exception as e:
                st.error(f"Error clearing ChromaDB data: {str(e)}")
        else:
            st.info("No ChromaDB data to clear.")

    if st.button("Clear Analysis Cache"):
        try:
            st.session_state.response_cache = {}
            st.session_state.answer_displayed = False
            st.session_state.last_question = ""
            st.success("Analysis cache cleared successfully!")
        except Exception as e:
            st.error(f"Error clearing cache: {str(e)}")


@st.cache_resource(show_spinner=False)
def initialize_models():
    model_id = "pratham0011/mistral_7b-instruct-research-paper"

    # Load tokenizer with optimized settings
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=False,  # Allow downloading if not cached
        use_fast=True           # Use faster tokenizer implementation
    )

    # Load model with enhanced memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use half precision
        low_cpu_mem_usage=True,     # Optimize memory usage
        offload_folder="offload",   # Offload to disk if needed
        device_map="auto",          # Let the library decide the best device mapping
        offload_state_dict=True,    # Offload state dict to CPU to save GPU memory
        use_cache=True              # Enable KV cache for faster inference
    )

    # Create a text generation pipeline with optimized settings
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        batch_size=1,               # Process one input at a time to reduce memory usage
        return_full_text=False      # Only return the generated text, not the prompt
    )

    # Create a wrapper function with optimized handling
    def mistral_llm(prompt):
        try:
            # Use a timeout to prevent hanging
            output = llm_pipeline(prompt)

            # Since we set return_full_text=False, we should get only the generated text
            generated_text = output[0]["generated_text"]

            # Clean up any remaining prompt text if needed
            if prompt in generated_text:
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text

            return response
        except Exception as e:
            # Provide a fallback response if model generation fails
            st.error(f"Model generation error: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request. Error details: {str(e)}"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return mistral_llm, embeddings

# Add a model loading state to prevent reloading
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Only load the model if it hasn't been loaded yet
if not st.session_state.model_loaded:
    with st.spinner("Loading Mistral 7B model... This may take a few minutes on first run."):
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)

        # Display staged loading messages for better user experience
        loading_message = st.empty()
        loading_message.info("Stage 1/3: Initializing model components...")
        progress_bar.progress(10)

        # Pre-download the model to disk cache if needed
        loading_message.info("Stage 2/3: Loading model into memory (this may take a while)...")
        progress_bar.progress(30)

        # Load models
        llm, embeddings = initialize_models()

        # Update progress and message
        loading_message.info("Stage 3/3: Finalizing model setup...")
        progress_bar.progress(90)

        # Mark as loaded in session state
        st.session_state.model_loaded = True

        # Final update
        progress_bar.progress(100)
        loading_message.success("✅ Model loaded successfully! Ready to analyze research papers.")
else:
    # If already loaded, just get the cached models
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

                # Generate answer using our Mistral model
                answer = llm(prompt)

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