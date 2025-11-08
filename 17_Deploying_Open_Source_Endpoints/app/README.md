## App package structure

This `app` package organizes LangGraph/LangChain agent graphs, shared state, model configuration, and tool integrations into focused modules. The goal is to keep each concern small and composable so you can mix and match graphs and capabilities without duplicating wiring.

### Layout

- `__init__.py`: Lightweight bootstrap that loads a local `.env` (for local dev) and exposes subpackages via `__all__`.
- `models.py`: Central place to construct chat LLM clients using Together AI's endpoints with consistent defaults. Graphs import `get_chat_model()` instead of re-creating clients.
- `state.py`: Shared `AgentState` schema used by graphs. Uses `add_messages` to safely accumulate messages across steps.
- `tools.py`: Aggregates third-party tools (Tavily) and local tools (RAG) into a single tool belt for easy binding to models.
- `rag.py`: Minimal Retrieval-Augmented Generation pipeline. Loads CSV files from `RAG_DATA_DIR`, chunks, embeds with Together AI's BAAI/bge-large-en-v1.5 model, stores in in-memory Qdrant, and exposes a `retrieve_information` Tool.
- `graphs/`: Collection of agent graphs that orchestrate model calls, tool execution, and data retrieval.
  - `simple_agent.py`: Smallest useful agent: model -> optional tools -> done.
  - `simple_graph.py`: Simple RAG graph that retrieves context and generates responses without tool calling.

### Why this structure

- **Separation of concerns**: Models, state, tools, and graphs live in dedicated modules. Each can evolve independently (swap models, add tools, change routing) with minimal cross-coupling.
- **Reusability**: `get_tool_belt()` and `get_chat_model()` can be reused across multiple graphs; `AgentState` standardizes message passing.
- **Testability**: Small, focused modules are easier to test in isolation (e.g., unit test the RAG tool without the agent graphs).
- **Extensibility**: Add a new tool or graph by creating a new module without touching existing ones, then import/bind where needed.

### Environment variables

- `TOGETHER_MODEL`: Controls which Together AI chat model to use (default: `meta-llama/Llama-3.3-70B-Instruct-Turbo`).
- `TOGETHER_API_KEY`: API key for Together AI services.
- `RAG_DATA_DIR`: Directory containing CSV files to index for the RAG tool (default: `data`).
- `TAVILY_API_KEY`: API key for Tavily search tool (optional).

### Typical usage

Graphs import from the shared modules:

```python
from app.models import get_chat_model
from app.tools import get_tool_belt
from app.state import AgentState
```

Then bind tools to the model and construct a `StateGraph` that routes between the agent node and a `ToolNode` for tool execution.

### Key differences from the OpenAI version

This implementation uses Together AI's open-source endpoints instead of OpenAI:

- **Embedding model**: `BAAI/bge-large-en-v1.5` via `TogetherEmbeddings`
- **Chat model**: `openai/gpt-oss-20b` (configurable) via `ChatTogether`
- **Data format**: Loads CSV files instead of PDFs for RAG pipeline

# Use

Run `uv run langgraph dev` in order to serve the graph in Langgraph studio.
