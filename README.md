# Multimodal RAG for Architectural Knowledge Management

A research prototype exploring how **Retrieval-Augmented Generation (RAG)** can make architectural project knowledge easier to discover and reuse across large design practices. Built for the **PE7046 Research Project** module of the MSc Computer Science with Artificial Intelligence at Northumbria University.

The system combines semantic search across **project text and imagery** with locally hosted language models to answer natural-language questions using retrieved project context. A Flask web interface presents answers alongside links to relevant architectural projects.

🎥 **[Watch the Project Demo (YouTube)](https://youtu.be/QoHz8BOowLI)**

📄 **[Read the Full Research Report (PDF)](report/w23051337-PE7046-Research-Project-Salvador-Navarro.pdf)**

## Main Features

- **Multimodal retrieval:** Separate ChromaDB collections for text, images and image captions, with configurable text-only, image-only and combined retrieval modes.
- **Document ingestion:** Utilities for collecting source material, extracting content from PDFs, preparing architectural project data and chunking text for vector indexing.
- **Semantic search:** Ollama text embeddings and OpenCLIP image embeddings to retrieve relevant project information.
- **Image understanding:** BLIP-based image caption generation and visual question answering (VQA) to incorporate visual information into retrieval.
- **Grounded answer generation:** Locally served LLMs via Ollama generate answers from retrieved context; optional query rewriting improves retrieval queries.
- **Retrieval refinement:** Configurable top-k selection, modality weighting and CrossEncoder reranking.
- **Web application:** Flask-based conversational interface with project thumbnails and project-detail pages linked to retrieved sources.
- **Evaluation and testing:** Dedicated modules for retrieval/generation evaluation, baseline and benchmark comparisons, profiling and automated tests.

## Technology Stack

| Area | Technologies |
| --- | --- |
| Backend / web | Python, Flask, SQLAlchemy, SQLite |
| LLM inference | Ollama, LangChain |
| Vector search | ChromaDB |
| Text and image embeddings | Ollama embeddings, OpenCLIP |
| Computer vision | BLIP, Hugging Face Transformers, PyTorch |
| Reranking | Sentence Transformers CrossEncoder |
| Document processing | PyMuPDF, LangChain text splitters |
| Testing | pytest |

## How It Works

1. **Prepare data:** Collect architectural project material and extract text and images from source documents.
2. **Index content:** Split text into chunks, generate embeddings and populate persistent ChromaDB collections for text, images and captions.
3. **Retrieve evidence:** Embed the user's question and search the selected modality collections; optionally rewrite the query and rerank retrieved candidates.
4. **Generate an answer:** Supply the retrieved context to a local LLM with instructions to answer from that evidence.
5. **Explore sources:** Display the answer and associated architectural projects through the Flask interface.

## Repository Structure

- `services/rag/` — RAG orchestration, retrieval, generation, embeddings, captions, VQA and reranking.
- `app/` — Flask routes, templates, UI assets and project database access.
- `utility/scraper/` — Source collection and PDF generation.
- `utility/db_setup/` — Project database setup and PDF extraction.
- `utility/data_analysis/` — Text and semantic data analysis.
- `utility/rag_evaluation/` — Evaluation metrics, benchmarks and profiling.
- `tests/` — Automated tests for core RAG components.
- `Makefile` — Environment, setup, execution and evaluation commands.

## Setup and Usage

The project uses a **Conda/Mamba** environment and requires **Ollama** to be installed and running. Model files and project data must be available locally. The primary environment targets CUDA; a separate CPU environment definition is provided but is not fully tested.

```bash
# Create the environment (Mamba required for this command)
make env-create

# Activate the environment
conda activate PE7046

# Prepare data and build the project/vector databases
make setup

# Launch the Flask application
make run
```

Additional commands include `make da` for data analysis, `make eval` for RAG evaluation and `pytest` for the test suite. See the `Makefile` for alternative setup and benchmarking commands.

## Research Context

The project investigates the opportunities and limitations of using **local, multimodal RAG** to support knowledge retrieval in architecture, including the contribution of image-based retrieval, query refinement and reranking. It is a **research prototype**, not a production deployment.

**Author:** Salvador Cristobal Navarro Perez  
**Supervisor:** Keerthana Jaganathan, Northumbria University  
**Second marker:** Abbas Mehrabidavoodabadi, Northumbria University  
**Module tutor:** Dr Ian Watson, Northumbria University
