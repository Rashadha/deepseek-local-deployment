# document_processing.py

# Import required libraries
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

# Function to process a PDF and create a retriever
def process_pdf(pdf_path):
    # Load the PDF using PDFPlumberLoader
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()

    # Split the text into semantically meaningful chunks
    text_splitter = SemanticChunker(HuggingFaceBgeEmbeddings())
    documents = text_splitter.split_documents(docs)

    # Initialize the embedding model for vector search
    embedder = HuggingFaceBgeEmbeddings()
    vector = FAISS.from_documents(documents, embedder)  # Store embeddings in FAISS

    return vector  # Return the vector database for retrieval
