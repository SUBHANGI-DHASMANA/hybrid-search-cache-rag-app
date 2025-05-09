# Research Paper Q&A Assistant

## Demo

https://github.com/user-attachments/assets/d07e9730-c22a-43a5-9757-c0a5097e8018

## Overview

The Research Paper Q&A Assistant is an AI-powered application designed to help researchers, students, and academics efficiently extract insights from scientific papers. By leveraging advanced natural language processing and retrieval techniques, the application enables users to ask questions about research papers and receive accurate, contextually relevant answers without having to read the entire document.

## Problem Statement

Researchers and students face significant challenges when trying to quickly understand and extract information from academic papers:

- **Information Overload**: The volume of published research is growing exponentially, making it impossible to thoroughly read every relevant paper
- **Time Constraints**: Researchers need to quickly extract key information from papers but lack efficient tools to do so
- **Complex Language**: Academic papers often use specialized terminology and complex sentence structures that can be difficult to parse
- **Cross-Referencing Challenges**: Understanding how findings relate to specific methodologies or data points within a paper requires significant cognitive effort
- **Limited Accessibility**: Traditional search methods within PDFs are keyword-based and miss semantic relationships

## Target Users

1. **Academic Researchers**: Professors, postdocs, and research scientists who need to quickly extract information from papers in their field
2. **Graduate Students**: Masters and PhD students conducting literature reviews or staying current with research
3. **Undergraduate Students**: Students working on research projects or trying to understand complex academic material
4. **Industry R&D Professionals**: Researchers in corporate settings who need to stay updated on academic advancements
5. **Journal Editors and Reviewers**: Professionals who need to efficiently assess research papers
6. **Research Librarians**: Information specialists who assist others in finding and understanding academic content

## Product Features

1. **Research Paper Processing**
   - Upload and process PDF research papers
   - Intelligent text extraction that preserves document structure
   - Automatic recognition of paper sections (abstract, methods, results, etc.)

2. **Advanced Retrieval System**
   - Hybrid search combining semantic (vector) and keyword (BM25) approaches
   - Context-aware retrieval that understands academic paper structure
   - Preservation of mathematical formulas and technical notation

3. **Academic-Focused Question Answering**
   - Specialized prompting for research paper analysis
   - Citation of specific paper sections in responses
   - Recognition of scientific terminology and concepts

4. **User Experience**
   - Intuitive interface designed for academic users
   - Clear presentation of answers with source citations
   - Response caching for improved performance

5. **Research Enhancement Tools**
   - Extraction of key findings, methods, and limitations
   - Identification of research gaps and future work
   - Highlighting of important citations and related work

## Technical Requirements

- **Backend**: Python-based with LangChain for orchestration
- **Frontend**: Streamlit for rapid development and iteration
- **Retrieval**: ChromaDB for vector storage, BM25 for keyword search
- **LLM Integration**: Hugging Face models (Mistral-7B-Instruct-v0.2 and sentence-transformers/all-MiniLM-L6-v2)
- **Data Processing**: PyPDF and custom extractors for academic papers
- **Deployment**: Containerized for easy deployment and scaling on Hugging Face Spaces

## Installation and Setup

1. Clone this repository:
   ```
   git clone https://github.com/SUBHANGI-DHASMANA/hybrid-search-cache-rag-app.git
   cd hybrid-search-cache-rag-app
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Troubleshooting

### Transformers Library Issues

If you encounter import errors with the transformers library, such as:
```
ImportError: cannot import name 'AutoTokenizer' from 'transformers'
```

You can fix this by running the provided script:
```
python fix_transformers.py
```

This script will:
1. Uninstall the current transformers library
2. Clean the pip cache
3. Install a specific version of transformers that is known to work
4. Install additional dependencies

Alternatively, you can manually reinstall the transformers library:
```
pip uninstall -y transformers
pip install --no-cache-dir transformers>=4.30.0,<4.40.0
```

### Memory Issues

If you encounter memory issues when loading the model, try:
1. Closing other applications to free up memory
2. Using a smaller model by modifying the `model_id` in `app.py`
3. Adjusting the model loading parameters in `app.py` to reduce memory usage
