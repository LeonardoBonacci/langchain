# LangChain PDF Summarizer

A Python application that uses LangChain to read PDF files and generate intelligent summaries using Ollama models locally.

## Features

- 📄 Read and parse PDF files
- 🤖 Generate concise summaries using LangChain and Ollama (runs locally!)
- 🔄 Two summarization methods:
  - **Map-Reduce**: Fast summarization for longer documents
  - **Refine**: Iterative refinement for higher quality summaries
- 📊 Automatic text chunking for large documents

## Installation

1. Install dependencies:
```bash
pip3 install -r requirements.txt
```

2. Make sure Ollama is installed and running:
```bash
# Check available models
ollama list

# The app uses qwen3:8b by default
# If you don't have it, pull it with:
ollama pull qwen3:8b
```

Note: The app will automatically download the GPT-2 tokenizer on first run for token counting.

## Usage

### Basic Usage (Map-Reduce method)
```bash
python3 pdf_summarizer.py path/to/your/document.pdf
```

### Using Refine method for better quality
```bash
python3 pdf_summarizer.py path/to/your/document.pdf refine
```

### Example Output
```
Loading PDF: identity-referee-declaration-dl26.pdf
Loaded 1 pages
Split into 2 chunks
Generating summary...

================================================================================
SUMMARY
================================================================================
[Your concise PDF summary appears here]
================================================================================
```

### Programmatic Usage
```python
from pdf_summarizer import summarize_pdf, summarize_pdf_with_refine

# Map-reduce method (faster)
summary = summarize_pdf("document.pdf")
print(summary)

# Refine method (higher quality)
summary = summarize_pdf_with_refine("document.pdf")
print(summary)
```

## How It Works

1. **Load PDF**: Uses PyPDFLoader to extract text from PDF files
2. **Split Text**: Divides the document into manageable chunks with overlap
3. **Summarize**: Uses LangChain's summarization chains with Ollama models (running locally)
4. **Output**: Returns a concise summary of the entire document

## Summarization Methods

- **map_reduce**: Summarizes each chunk independently, then combines them. Faster for large documents.
- **refine**: Iteratively refines the summary by processing chunks sequentially. Better quality but slower.

## Requirements

- Python 3.8+
- Ollama installed and running locally
- qwen3:8b model (recommended) or llama3.2:latest (or any other Ollama model)
- Internet connection on first run (to download GPT-2 tokenizer)
- See requirements.txt for all package dependencies

## License

See LICENSE file for details.