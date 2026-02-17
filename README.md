# LangChain PDF Summarizer & Multi-Agent Round Table

A Python application collection using LangChain and autogen to:
1. Read PDF files and generate intelligent summaries using Ollama models locally
2. Simulate round table discussions with historical AI agents

## Features

### PDF Summarizer
- 📄 Read and parse PDF files
- 🤖 Generate concise summaries using LangChain and Ollama (runs locally!)
- 🔄 Two summarization methods:
  - **Map-Reduce**: Fast summarization for longer documents
  - **Refine**: Iterative refinement for higher quality summaries
- 📊 Automatic text chunking for large documents

### Round Table Discussion
- 🗣️ Multi-agent conversations with AI representations of historical figures
- 👥 Five agents: Albert Einstein, Leonardo da Vinci, Marie Curie, Socrates, and Ada Lovelace
- 💬 Dynamic topic-based discussions using local LLMs
- ✨ No API keys required - runs completely locally with Ollama

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

### PDF Summarizer

#### Basic Usage (Map-Reduce method)
```bash
python3 pdf_summarizer.py path/to/your/document.pdf
```

#### Using Refine method for better quality
```bash
python3 pdf_summarizer.py path/to/your/document.pdf refine
```

#### Example Output
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

#### Programmatic Usage
```python
from pdf_summarizer import summarize_pdf, summarize_pdf_with_refine

# Map-reduce method (faster)
summary = summarize_pdf("document.pdf")
print(summary)

# Refine method (higher quality)
summary = summarize_pdf_with_refine("document.pdf")
print(summary)
```

### Round Table Discussion

Run a multi-agent discussion with historical figures:
```bash
python3 auto_gen_round_table.py
```

The script will:
1. Generate a round table discussion on "The future of humanity shaped by technology and science"
2. Involve 5 AI agents representing historical figures
3. Produce up to 10 turns of conversation before terminating
4. Display all contributions from each agent

Example output shows agents engaging thoughtfully on topics, building on each other's ideas.

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
- Available models: llama3.2:latest (recommended), qwen3:8b, or any other Ollama model
- Internet connection on first run (to download GPT-2 tokenizer for PDF summarizer)
- See requirements.txt for all package dependencies

## Configuration

### Models
- **PDF Summarizer**: Uses `qwen3:8b` by default
- **Round Table Discussion**: Uses `llama3.2:latest` by default (better for text generation)

To use different models, edit the respective Python files and update the `model="..."` parameter.

### Installing Models with Ollama
```bash
# Pull llama3.2 (smaller, faster)
ollama pull llama3.2

# Pull qwen3:8b (powerful)
ollama pull qwen3:8b

# List installed models
ollama list
```

## License

See LICENSE file for details.