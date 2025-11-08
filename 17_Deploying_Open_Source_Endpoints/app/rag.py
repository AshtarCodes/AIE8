"""Retrieval-Augmented Generation (RAG) utilities and tool.

This module builds an in-memory RAG pipeline that:
- Loads CSV documents from `RAG_DATA_DIR` (default: "data").
- Splits documents into chunks using RecursiveCharacterTextSplitter.
- Embeds chunks with Together AI's BAAI/bge-large-en-v1.5 and stores vectors in an in-memory Qdrant store.
- Exposes a LangChain Tool `retrieve_information` that retrieves relevant
  context and generates a response constrained to that context.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated, List

from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_together import ChatTogether, TogetherEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict


class _RAGState(TypedDict):
    """State schema for the simple two-step RAG graph: retrieve then generate."""
    question: str
    context: List[Document]
    response: str


def _build_rag_graph(data_dir: str) -> "CompiledGraph":
    """Construct and compile a minimal RAG graph.

    Steps:
    1) Load CSV files from `data_dir` (best-effort).
    2) Split documents into chunks.
    3) Create embeddings with Together AI and an in-memory Qdrant vector store retriever.
    4) Define a chat prompt and generation model.
    5) Wire a two-node graph: retrieve -> generate.
    """
    # Load CSV files from data directory
    documents = []
    try:
        for file in os.listdir(data_dir):
            if file.endswith(".csv"):
                with open(os.path.join(data_dir, file), "r") as f:
                    documents.append(Document(page_content=f.read(), metadata={"source": file}))
    except Exception:
        documents = []

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents) if documents else []

    # Create embedding model using Together AI's BAAI/bge-large-en-v1.5 model
    embedding_model = TogetherEmbeddings(
        model="BAAI/bge-large-en-v1.5"
    )

    # Embeddings and vector store (in-memory Qdrant)
    qdrant_vectorstore = Qdrant.from_documents(
        documents=chunks, embedding=embedding_model, location=":memory:"
    )
    retriever = qdrant_vectorstore.as_retriever()

    # Prompt and model
    human_template = (
        "\n#CONTEXT:\n{context}\n\nQUERY:\n{query}\n\n"
        "Use the provided context to answer the provided user query. "
        "Only use the provided context to answer the query. If you do not know the answer, "
        "or it's not contained in the provided context respond with \"I don't know\""
    )
    chat_prompt = ChatPromptTemplate.from_messages([("human", human_template)])
    
    # Create together client using ChatTogether
    generator_llm = ChatTogether(
        model=os.environ.get("TOGETHER_MODEL", "openai/gpt-oss-20b"),
        temperature=0.7,
    )

    def retrieve(state: _RAGState) -> _RAGState:
        retrieved_docs = retriever.invoke(state["question"]) if retriever else []
        return {"context": retrieved_docs}  # type: ignore

    def generate(state: _RAGState) -> _RAGState:
        generator_chain = chat_prompt | generator_llm | StrOutputParser()
        response_text = generator_chain.invoke(
            {"query": state["question"], "context": state.get("context", [])}
        )
        return {"response": response_text}  # type: ignore

    graph_builder = StateGraph(_RAGState)
    graph_builder = graph_builder.add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


@lru_cache(maxsize=1)
def _get_rag_graph():
    """Return a cached compiled RAG graph built from RAG_DATA_DIR."""
    data_dir = os.environ.get("RAG_DATA_DIR", "data")
    return _build_rag_graph(data_dir)


@tool
def retrieve_information(
    query: Annotated[str, "query to ask the retrieve information tool"]
):
    """Use Retrieval Augmented Generation to retrieve information about AIE8 projects and domains."""
    graph = _get_rag_graph()
    result = graph.invoke({"question": query})
    # Prefer returning the response string if available
    if isinstance(result, dict) and "response" in result:
        return result["response"]
    return result

