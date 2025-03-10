# question_answering.py

# Import required libraries
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from document_processing import process_pdf

# Function to set up a QA system based on the processed document
def setup_qa_system(pdf_path):
    vector = process_pdf(pdf_path)  # Process the PDF and get the vector store
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})  # Retrieve top 3 similar documents

    # Load the DeepSeek R1 model from Ollama
    llm = Ollama(model="deepseek-r1:1.5b")

    # Define the prompt template for answering questions using document context
    prompt = """
    Use the below context to answer the document-based question.
    Context: {context}
    Question: {question}
    Answer:"""

    QA_PROMPT = PromptTemplate.from_template(prompt)

    # Define the LLM chain to process the question-answering task
    llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
    combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    # Create the retrieval-based QA pipeline
    qa_pipeline = RetrievalQA(combine_documents_chain=combine_documents_chain, retriever=retriever)

    return qa_pipeline

if __name__ == "__main__":
    # Get PDF file path from user
    pdf_path = input("Enter the path to the PDF file: ").strip()
    
    # Set up the QA pipeline
    qa_pipeline = setup_qa_system(pdf_path)

    while True:
        # Get user input
        question = input("\nAsk a question about the document (or type 'exit' to quit): ").strip()
        
        if question.lower() == "exit":
            print("Exiting...")
            break
        
        # Get the answer from the retrieval-based QA pipeline
        response = qa_pipeline(question)['result']
        
        # Display the response
        print("\nResponse:\n", response)
