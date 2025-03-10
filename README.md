```markdown
# Document Processing and Question Answering System

## Overview

This project provides a system for processing PDF documents and setting up a question-answering (QA) pipeline using a retriever and a language model. It consists of two main Python scripts:

1. `document_processing.py` - Processes a PDF file, extracts text, splits it into meaningful chunks, and stores it in a FAISS vector database.
2. `question_answering.py` - Sets up a QA system that retrieves relevant document chunks and answers questions using a language model.

## Features

- **PDF Processing:** Extracts text from PDF documents.
- **Semantic Text Chunking:** Uses `SemanticChunker` to split text into meaningful chunks.
- **Vector Search:** Stores document embeddings in a FAISS vector store.
- **Question Answering:** Uses an `Ollama` language model to answer questions based on retrieved document context.

## Installation

### Prerequisites

Ensure you have Python installed (>= 3.8). Install dependencies using pip:

```bash
pip install langchain langchain-community langchain-experimental pdfplumber faiss-cpu
```

### Setting Up Ollama

You need to install Ollama and load the DeepSeek R1 model:

```bash
pip install ollama
ollama pull deepseek-r1:1.5b
```

## Usage

### 1. Processing a PDF

Use `document_processing.py` to process a PDF file:

```python
from document_processing import process_pdf

vector_store = process_pdf("path/to/document.pdf")
```

### 2. Running the QA System

Run `question_answering.py` interactively:

```bash
python question_answering.py
```

You will be prompted to enter the PDF path and can then ask questions about its content.

### Example Usage:

```bash
Enter the path to the PDF file: example.pdf

Ask a question about the document (or type 'exit' to quit): What is the main topic?
Response:
The document discusses...
```

## File Descriptions

### `document_processing.py`
- Loads a PDF file and extracts text using `PDFPlumberLoader`.
- Splits text into semantically meaningful chunks using `SemanticChunker`.
- Embeds chunks using `HuggingFaceBgeEmbeddings` and stores them in FAISS.

### `question_answering.py`
- Loads the vector store and retrieves similar document chunks.
- Uses `Ollama`'s `deepseek-r1:1.5b` model to generate responses.
- Implements a retrieval-based QA system using `RetrievalQA`.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests.

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, reach out via GitHub Issues.
```
