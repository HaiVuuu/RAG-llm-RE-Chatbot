# RAG-llm-RE-Chatbot (RAG-VietLex)

An advanced real estate and corporate law search assistant leveraging Artificial Intelligence (AI) and Retrieval-Augmented Generation (RAG) to optimize data retrieval and reinforce the accuracy of domain-specific answers. This framework is specifically designed to handle complex, multi-hop queries regarding **Vietnamese Enterprise and Real Estate Law**.

## 📁 Repository Structure
The repository consists of 4 core Jupyter Notebooks, divided systematically by development, deployment, and evaluation phases:

1. **`qdrant-inject-final.ipynb`**: Responsible for data preprocessing, rule-based hierarchical chunking (article/clause-level), and injecting the entire legal corpus into the Qdrant vector database.
2. **`test-rag-final.ipynb`**: The sandbox environment for configuring, optimizing, and testing the entire RAG pipeline—integrating retrieval mechanisms, reranking thresholds, and LLM inference parameters.
3. **`gradio.ipynb`**: Handles production-level deployment by creating an intuitive, web-based chat interface using Gradio with real-time token streaming capabilities.
4. **`evaluation_final.ipynb`**: Runs comprehensive performance evaluation benchmarks, including automated NLP metrics (ROUGE, BERTScore, Perplexity) and the multi-judge LLM-as-a-Judge cross-evaluation protocol.

## 🌟 Key Features
* **Hybrid Retrieval Pipeline:** Merges Dense semantic retrieval (`bkai-foundation-models/vietnamese-bi-encoder`) with Sparse lexical matching (BM25 via `FastEmbedSparse`) to capture both conceptual intent and exact statutory keywords.
* **Cross-Encoder Reranking:** Integrates the state-of-the-art `BAAI/bge-reranker-v2-m3` model to aggressively filter out retrieval noise, mitigate semantic drift, and eliminate "knowledge override" hallucinations.
* **Optimized Local LLM Execution:** Powered by Qwen 2.5-7B, fine-tuned locally using Parameter-Efficient Fine-Tuning (QLoRA) and quantized to 4-bit GGUF format, ensuring high-fidelity statutory reasoning on consumer-grade hardware.
* **Interactive Streaming UI:** Built natively with Gradio to support seamless multi-turn conversations and smooth, low-latency streaming text generation.

## ⚙️ Tech Stack
* **Vector Database:** Qdrant
* **Orchestration Framework:** LangChain
* **LLM Inference Engine:** `llama-cpp-python`
* **Frontend Interface:** Gradio
* **Embedding & Reranker Models:** HuggingFace Transformers (`sentence-transformers`)

## 🚀 Prerequisites & System Requirements
To replicate and run this project, ensure you meet the following requirements:
1. **Hardware:** A Kaggle workspace or equivalent environment equipped with an NVIDIA Tesla T4 GPU (16GB VRAM).
2. **Database State:** The chunked legal corpus (containing over 380,000 chunks) must be fully injected and accessible in the Qdrant local storage directory (e.g., `/kaggle/working/qdrant_writable_db`).
3. **Authentication:** A valid Hugging Face Access Token securely stored in Kaggle's `UserSecretsClient` to authorize the automated download of foundation models and encoders.

## 🛠️ Step-by-Step Execution Guide
1. **Data Initialization:** Execute **`qdrant-inject-final.ipynb`** first to ensure your vector database is populated, indexed, and ready for queries.
2. **Environment Setup:** Open **`gradio.ipynb`** (or `test-rag-final.ipynb`) inside your Kaggle workspace.
3. **Database Mapping:** Verify that the populated Qdrant database folder is correctly located or copied into the active working directory (`/kaggle/working/`).
4. **Dependency Installation:** Run the initialization cells to install all required libraries (`langchain`, `qdrant-client`, `llama-cpp-python`, `gradio`).
5. **App Launch:** Run the final Gradio cell to initiate the public share link and interact with the legal assistant web application.

## 📊 Evaluation & Benchmarking
To perform comprehensive quality checks using **`evaluation_final.ipynb`**, the system requires:
* A held-out test set consisting of synthetic legal question-answer pairs to calculate **ROUGE-1/2/L**, **BERTScore (F1)**, and **Mean Perplexity** scores.
* API keys for industry-leading frontier models (GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet, and Grok) to execute the **LLM-as-a-Judge** protocol, scoring answers on a strict 1–5 scale for **Faithfulness** and **Answer Relevance**.
