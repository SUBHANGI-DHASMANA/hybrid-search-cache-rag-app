# Hybrid Search Cache RAG Application

A Streamlit-based application that processes PDF documents, indexes them using hybrid retrieval (ChromaDB + BM25), and provides cached responses to user queries for efficient question-answering.

## Features

- **PDF Processing**: Upload and extract text from PDF documents
- **Hybrid Retrieval**: Combines vector embeddings (ChromaDB) and BM25 for comprehensive document retrieval
- **Cached Responses**: Smart caching mechanism to avoid redundant LLM calls
- **Context-Aware QA**: Ask questions about your document content with relevant context
- **Streamlit UI**: User-friendly interface for easy interaction

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/hybrid-search-cache-rag-app.git
   ```

   ```
   cd hybrid-search-cache-rag-app
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```