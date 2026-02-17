"""
LangChain PDF Summarizer
Reads a PDF file and generates a summary using LangChain and Ollama.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate


def summarize_pdf(pdf_path: str, model_name: str = "qwen3:8b") -> str:
    """
    Summarize a PDF file using LangChain with Ollama.
    
    Args:
        pdf_path: Path to the PDF file
        model_name: Ollama model to use (default: qwen3:8b)
    
    Returns:
        Summary of the PDF content
    """
    
    # Load PDF
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    # Initialize LLM
    llm = ChatOllama(
        model=model_name,
        temperature=0
    )
    
    # Create summarization chain
    # Using map_reduce for longer documents
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        verbose=True
    )
    
    # Generate summary
    print("Generating summary...")
    summary = chain.run(chunks)
    
    return summary


def summarize_pdf_with_refine(pdf_path: str, model_name: str = "qwen3:8b") -> str:
    """
    Summarize a PDF using the refine method for better quality.
    
    Args:
        pdf_path: Path to the PDF file
        model_name: Ollama model to use (default: qwen3:8b)
    
    Returns:
        Summary of the PDF content
    """
    
    # Load PDF
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    # Initialize LLM
    llm = ChatOllama(
        model=model_name,
        temperature=0
    )
    
    # Create custom prompts
    prompt_template = """Write a concise summary of the following:

{text}

CONCISE SUMMARY:"""
    
    refine_template = """Your job is to produce a final summary.
We have provided an existing summary up to a certain point: {existing_answer}
We have the opportunity to refine the existing summary (only if needed) with some more context below.
------------
{text}
------------
Given the new context, refine the original summary. If the context isn't useful, return the original summary.
REFINED SUMMARY:"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    refine_prompt = PromptTemplate.from_template(refine_template)
    
    # Create chain with refine
    chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        verbose=True
    )
    
    # Generate summary
    print("Generating summary with refine method...")
    summary = chain.run(chunks)
    
    return summary


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_summarizer.py <path_to_pdf> [method]")
        print("method: 'map_reduce' (default) or 'refine'")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "map_reduce"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    try:
        if method == "refine":
            summary = summarize_pdf_with_refine(pdf_path)
        else:
            summary = summarize_pdf(pdf_path)
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(summary)
        print("="*80)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
