# import embedding model and together client
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv("../.env")

# create embedding model using Together AI's endpoint
# Together AI provides embeddings via OpenAI-compatible API
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # openai_api_base="https://api.together.xyz/v1",
    # openai_api_key=os.getenv("TOGETHER_API_KEY"),
)

# create together client using ChatOpenAI with Together AI's endpoint
llm_client = ChatOpenAI(
    model="openai/gpt-oss-20b",
    base_url="https://api.together.xyz/v1",
    api_key=os.getenv("TOGETHER_API_KEY"),
    temperature=0.7,
)

# use chroma in memory to store and query documents
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
# import the csv from ./data/*.csv and split into documents
documents = []
for file in os.listdir("./data"):
    if file.endswith(".csv"):
        with open(os.path.join("./data", file), "r") as f:
            documents.append(Document(page_content=f.read(), metadata={"source": file}))

#  split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# create chroma in memory vector store
chroma = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db",
)

# create retriever
retriever = chroma.as_retriever()

# create prompt
prompt = ChatPromptTemplate.from_template(
    template="""
    You are a helpful assistant that can answer questions about the following documents:
    {context}
    Question: {question}
    Answer:
    """
)

# creat langgraph state
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, TypedDict
from langchain_core.output_parsers import StrOutputParser

class _RAGState(TypedDict):
    """State schema for the simple two-step RAG graph: retrieve then generate."""
    question: str
    context: List[Document]
    response: str

def retrieve(state: _RAGState) -> _RAGState:
    retrieved_docs = retriever.invoke(state["question"]) if retriever else []
    return {"context": retrieved_docs}  # type: ignore

def generate(state: _RAGState) -> _RAGState:
    generator_chain = prompt | llm_client | StrOutputParser()
    response_text = generator_chain.invoke(
        {"question": state["question"], "context": state.get("context", [])}
    )
    return {"response": response_text}  # type: ignore

graph_builder = StateGraph(_RAGState)
graph_builder = graph_builder.add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")

# run chain
graph = graph_builder.compile()
response = graph.invoke({"question": "What is the most common primary domain for the AIMakerspace projects?"})
print(response["response"])