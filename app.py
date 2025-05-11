import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import time
import shutil
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Create directories if they don't exist
os.makedirs("offload", exist_ok=True)

from data_ingestion import process_uploaded_file, load_existing_index
from chroma_store import update_retriever_weights

st.title("Research Paper Q&A Assistant")
st.caption("Powered by Mistral 7B Instruct Research Paper model")

# Initialize all session state variables at the beginning
if 'response_cache' not in st.session_state:
    st.session_state.response_cache = {}
if 'answer_displayed' not in st.session_state:
    st.session_state.answer_displayed = False
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# UI Columns for buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Clear Research Paper Database"):
        chroma_dir = os.path.join(os.getcwd(), "chroma_db")
        if os.path.exists(chroma_dir):
            try:
                shutil.rmtree(chroma_dir)
                st.session_state.retriever = None
                st.success("Research paper database cleared successfully!")
            except Exception as e:
                logger.error(f"Error clearing ChromaDB data: {str(e)}")
                st.error(f"Error clearing database: {str(e)}")
        else:
            st.info("No ChromaDB data to clear.")

with col2:
    if st.button("Clear Analysis Cache"):
        try:
            st.session_state.response_cache = {}
            st.session_state.answer_displayed = False
            st.session_state.last_question = ""
            st.success("Analysis cache cleared successfully!")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            st.error(f"Error clearing cache: {str(e)}")

@st.cache_resource(show_spinner=False)
def initialize_models():
    try:
        model_id = "pratham0011/mistral_7b-instruct-research-paper"
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            offload_folder="offload",
            device_map="auto"
        )

        # Create pipeline
        llm_pipeline = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_new_tokens=512, 
            do_sample=True,
            temperature=0.7
        )

        def mistral_llm(prompt):
            try:
                output = llm_pipeline(prompt)
                generated_text = output[0]["generated_text"]
                
                # Extract only the new generated text (remove prompt)
                if prompt in generated_text:
                    response = generated_text[len(prompt):].strip()
                else:
                    response = generated_text

                return response
            except torch.cuda.OutOfMemoryError:
                error_msg = "The model ran out of memory. Try asking a shorter question or splitting your query."
                logger.error(error_msg)
                return error_msg
            except Exception as e:
                error_msg = f"Model generation error: {str(e)}"
                logger.error(error_msg)
                return f"I apologize, but I encountered an error while processing your request. Error details: {str(e)}"

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        return mistral_llm, embeddings

    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        st.error(f"Failed to initialize models: {str(e)}")
        raise e

# Model loading section
if not st.session_state.model_loaded:
    with st.spinner("Loading Mistral 7B model... This may take a few minutes on first run."):
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)

        loading_message = st.empty()
        loading_message.info("Stage 1/3: Initializing model components...")
        progress_bar.progress(10)

        try:
            loading_message.info("Stage 2/3: Loading model into memory (this may take a while)...")
            progress_bar.progress(30)

            llm, embeddings = initialize_models()

            loading_message.info("Stage 3/3: Finalizing model setup...")
            progress_bar.progress(90)

            st.session_state.model_loaded = True
            progress_bar.progress(100)
            loading_message.success("✅ Model loaded successfully! Ready to analyze research papers.")
        except Exception as e:
            loading_message.error(f"❌ Model loading failed: {str(e)}")
            st.stop()
else:
    llm, embeddings = initialize_models()

# Main UI
st.markdown("### Upload a research paper to analyze and ask questions about its content.")
st.markdown("This tool helps you extract insights, understand methodologies, and explore findings from academic papers.")

pdf_file = st.file_uploader("Upload your research paper (PDF format)", type=['pdf'])

# Check for existing papers
try:
    with st.spinner("Checking for previously analyzed research papers..."):
        existing_retriever = load_existing_index()
        if existing_retriever:
            st.session_state.retriever = existing_retriever
            st.success("Found previously analyzed research papers!")
except Exception as e:
    logger.error(f"Error checking for previous research papers: {str(e)}")
    st.error(f"Error checking for previous research papers: {str(e)}")

# Process uploaded file
if pdf_file is not None:
    with st.spinner("Processing research paper and extracting content for analysis..."):
        try:
            retriever, tmp_file_path = process_uploaded_file(pdf_file)
            
            # Update retriever weights
            bm25_weight = 0.5
            vector_weight = 0.5
            retriever = update_retriever_weights(retriever, bm25_weight, vector_weight)
            
            st.session_state.retriever = retriever
            st.success("Research paper processed and ready for analysis!")
            st.session_state.answer_displayed = False
            st.session_state.last_question = ""
            
        except Exception as e:
            logger.error(f"Error processing research paper: {str(e)}")
            st.error(f"Error processing research paper: {str(e)}")
        finally:
            if 'tmp_file_path' in locals() and tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

# Question answering section
if st.session_state.retriever is not None:
    # Create columns for the question input and Ask button
    question_col, button_col = st.columns([4, 1])

    with question_col:
        question = st.text_input(
            "Ask a question about the research paper:",
            placeholder="e.g., 'What methodology was used?', 'What are the key findings?'"
        )

    with button_col:
        ask_button = st.button(
            "Ask",
            key="ask_button",
            use_container_width=True,
            disabled=st.session_state.get('processing', False))
    
    if (question and ask_button) or (question and not st.session_state.answer_displayed):
        try:
            st.session_state.processing = True
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_display = st.empty()

            start_time = time.time()

            # Phase 1: Retrieving documents
            phase1_start = time.time()
            status_text.text("Retrieving relevant documents...")
            progress_bar.progress(10)

            docs = st.session_state.retriever.get_relevant_documents(question)
            phase1_end = time.time()
            phase1_time = phase1_end - phase1_start
            progress_bar.progress(40)

            time_display.text(f"⏱️ Retrieval time: {phase1_time:.2f} seconds")

            # Phase 2: Generating answer
            phase2_start = time.time()
            status_text.text("Generating answer...")
            cache_key = question.strip().lower()

            if cache_key in st.session_state.response_cache:
                result = st.session_state.response_cache[cache_key]
                answer = result['answer']
                docs = result['source_documents']
            else:
                context = "\n\n".join([doc.page_content for doc in docs[:4]])
                prompt = f"""You are a research assistant analyzing academic papers. Use the following excerpts from a research paper to answer the question.

Paper excerpts:
{context}

Question: {question}

Provide a comprehensive answer based strictly on the paper content. Include:
1. Direct quotes or paraphrases from the text (cite page numbers if available)
2. Key findings related to the question
3. Methodologies used if relevant
4. Any limitations mentioned

If the question cannot be answered from the excerpts, respond with:
"I could not find sufficient information in the paper to answer this question. The relevant sections may not have been extracted or the question may require information not present in these excerpts."

Answer in clear academic prose, using bullet points for multiple findings when appropriate.

Answer:"""

                answer = llm(prompt)
                st.session_state.response_cache[cache_key] = {
                    'answer': answer,
                    'source_documents': docs
                }

            phase2_end = time.time()
            phase2_time = phase2_end - phase2_start

            total_time = time.time() - start_time

            progress_bar.progress(100)
            status_text.text("Done!")
            time_display.text(f"⏱️ Total time: {total_time:.2f} seconds | Retrieval: {phase1_time:.2f}s | Generation: {phase2_time:.2f}s")
            
            # Display results
            st.write("### Research Analysis:")
            st.markdown(answer)

            st.write("### Source Information:")
            st.info("The analysis above is based on the uploaded research paper and the specific sections referenced below.")

            with st.expander("View Detailed Paper Excerpts"):
                for i, doc in enumerate(docs[:4]):
                    page_num = doc.metadata.get('page', i+1) if hasattr(doc, 'metadata') and doc.metadata else i+1
                    st.markdown(f"**Excerpt {i+1} (Page {page_num})**")
                    
                    # Improved text display with better formatting
                    st.markdown("```\n" + doc.page_content + "\n```")
                    st.markdown("---")

            st.session_state.answer_displayed = True
            st.session_state.last_question = question
            
        except Exception as e:
            logger.error(f"Error during question answering: {str(e)}")
            st.error(f"An error occurred while processing your question: {str(e)}")
        finally:
            st.session_state.processing = False

# Clean up on app exit
def cleanup():
    if os.path.exists("offload"):
        try:
            shutil.rmtree("offload")
        except Exception as e:
            logger.warning(f"Could not clean up offload directory: {str(e)}")

import atexit
atexit.register(cleanup)