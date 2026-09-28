# 🤗 HuggingRAG: Zero-Cost RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from your documents using **free, open-source Hugging Face models**. No paid API keys are needed. It has conversational memory and a Streamlit chat UI, and it lets users upload their own `.txt` files on the fly.

<!-- Add a screenshot of your app here: -->
<!-- ![HuggingRAG demo](screenshot.png) -->

## Features

- **Free end-to-end**: local embeddings plus a free-tier Hugging Face chat model
- **Grounded answers**: the model answers only from retrieved context and says so when the information isn't there
- **History-aware chat**: follow-ups like *"When did that deal close?"* are rewritten into standalone search queries
- **Live document upload**: users can add `.txt` files from the sidebar
- **Private uploads**: uploaded files go into a per-session, in-memory vector store and are never written to the shared database
- **Hybrid search (notebook demo)**: vector search combined with BM25 keyword search

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (runs locally) |
| LLM | `Qwen/Qwen2.5-7B-Instruct` via the Hugging Face Inference API |
| Vector database | ChromaDB |
| Keyword search | BM25 (`rank_bm25`) |
| UI | Streamlit |

## How It Works

```
User question
     │
     ▼
Rewrite with chat history (standalone query)
     │
     ▼
Retrieve top chunks ──► core_db (shared docs, persistent)
     │              └─► session_db (user uploads, in-memory)
     ▼
Build prompt: question + retrieved context
     │
     ▼
Qwen2.5-7B-Instruct generates a grounded answer
```

1. Documents are split into 1000-character chunks and embedded with MiniLM.
2. Embeddings are stored in ChromaDB (`db/chroma_db`) for the shared knowledge base.
3. Each user session gets its own in-memory collection, so uploads are isolated and disappear when the session ends.
4. On each question, chunks are retrieved from both stores and passed to the LLM as context.

## Project Structure

```
HuggingRAG/
├── app.py              # Streamlit app (retrieval + generation + upload)
├── docs/               # Source documents (.txt)
├── db/chroma_db/       # Pre-built vector store
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/darshan320/HuggingRAG.git
cd HuggingRAG
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 3. Add your Hugging Face token
Create a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Read access is enough), then create a `.env` file in the project root:
```
HF_TOKEN=hf_your_token_here
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

### 4. Run the app
```bash
streamlit run app.py
```
It opens at `http://localhost:8501`.

## Usage

- Ask questions about the built-in documents (Google, Microsoft, Nvidia, SpaceX, Tesla).
- Upload your own `.txt` file from the sidebar and ask questions about it.
- Ask follow-up questions and the bot will remember the conversation.
- Use **Clear chat history** to reset the conversation.

## Deployment

The app is deployed on **Streamlit Community Cloud**. If you deploy your own copy, add `HF_TOKEN` and `HUGGINGFACEHUB_API_TOKEN` under the app's **Secrets** settings instead of committing a `.env` file.

## Notes and Limitations

- The free Hugging Face Inference API can be rate-limited or slow at busy times, and model availability on the free tier can change. If the chat model stops responding, swap `repo_id` in `app.py` for another chat-capable model.
- Only `.txt` files are supported at the moment.
- Answer quality depends on retrieval. If a fact isn't in the top retrieved chunks, the bot will say it doesn't have enough information.

## Roadmap

- [ ] Multi-modal ingestion (PDF tables and images)
- [ ] Hybrid (vector + BM25) retrieval inside the Streamlit app
- [ ] PDF upload support
- [ ] Source citations shown alongside answers

## Acknowledgements

Built while learning from Harish Neel's [rag-for-beginners](https://github.com/harishneel1/rag-for-beginners), with the OpenAI components replaced by Hugging Face models and a Streamlit UI plus upload feature added.

## License

MIT