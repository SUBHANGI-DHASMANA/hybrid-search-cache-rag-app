# Research Paper Q&A Assistant

This Streamlit application helps researchers, students, and academics efficiently extract insights from scientific papers. Upload a PDF research paper and ask questions about its content to get accurate, contextually relevant answers.

## Features

- **Upload and Process PDFs**: Easily upload research papers in PDF format
- **Ask Questions**: Query the paper about methodologies, findings, limitations, and more
- **Hybrid Search**: Combines semantic (vector) and keyword (BM25) approaches for better retrieval
- **Response Caching**: Saves previous answers for faster responses
- **Academic-Focused**: Specialized for research paper analysis with proper citations

## How to Use

1. Upload a research paper in PDF format using the file uploader
2. Wait for the paper to be processed (this may take a few moments)
3. Type your question in the text box (e.g., "What methodology was used?", "What are the key findings?")
4. Click "Ask" to get your answer
5. View the source excerpts to see where the information came from

## Models

This application uses Hugging Face models:
- **LLM**: mistralai/Mistral-7B-Instruct-v0.2 for generating answers
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 for semantic search

## Limitations

- The quality of answers depends on the clarity and structure of the uploaded paper
- Very large papers may take longer to process
- Complex tables, figures, and mathematical notations may not be perfectly preserved

## Feedback

If you encounter any issues or have suggestions for improvement, please open an issue on the GitHub repository.
