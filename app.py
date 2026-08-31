import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="HuggingRAG Chatbot", page_icon="🤗")
st.title("🤗 HuggingRAG — Ask Me Anything")
st.caption("A free, open-source RAG chatbot powered by Hugging Face models")

# ─────────────────────────────────────────────
# Cache the heavy resources so they load only once
# ─────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_model():
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="conversational",
        provider="auto",
        temperature=0.3,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=llm_endpoint)


embeddings = load_embeddings()
model = load_model()

# ─────────────────────────────────────────────
# Vector store: kept in session_state so uploads persist during the session
# ─────────────────────────────────────────────
if "db" not in st.session_state:
    st.session_state.db = Chroma(
        persist_directory="db/chroma_db",
        embedding_function=embeddings,
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "display_history" not in st.session_state:
    st.session_state.display_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ─────────────────────────────────────────────
# Sidebar: file upload
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Add your own documents")
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file is not None and uploaded_file.name not in st.session_state.uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Save to a temp file so TextLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # Load and chunk
            loader = TextLoader(tmp_path, encoding="utf-8")
            documents = loader.load()

            # Fix metadata to show the real filename instead of the temp path
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            chunks = splitter.split_documents(documents)

            # Add directly into the existing vector store
            st.session_state.db.add_documents(chunks)

            st.session_state.uploaded_files.append(uploaded_file.name)
            os.remove(tmp_path)

        st.success(f"Added {uploaded_file.name} ({len(chunks)} chunks)")

    if st.session_state.uploaded_files:
        st.markdown("**Uploaded this session:**")
        for f in st.session_state.uploaded_files:
            st.markdown(f"- {f}")

    if st.button("🗑️ Clear chat history"):
        st.session_state.chat_history = []
        st.session_state.display_history = []
        st.rerun()

# ─────────────────────────────────────────────
# Render past chat messages
# ─────────────────────────────────────────────
for role, text in st.session_state.display_history:
    with st.chat_message(role):
        st.markdown(text)

# ─────────────────────────────────────────────
# Core RAG logic
# ─────────────────────────────────────────────
def ask_question(user_question):
    chat_history = st.session_state.chat_history
    db = st.session_state.db

    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        result = model.invoke(messages)
        search_question = result.content.strip()
    else:
        search_question = user_question

    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    result = model.invoke(messages)
    answer = result.content

    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.append(AIMessage(content=answer))

    return answer

# ─────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────
user_input = st.chat_input("Ask a question about the documents...")

if user_input:
    st.session_state.display_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_question(user_input)
            st.markdown(answer)

    st.session_state.display_history.append(("assistant", answer))


# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"