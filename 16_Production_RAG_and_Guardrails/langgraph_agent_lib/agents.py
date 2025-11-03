"""LangGraph agent integration with production features."""

from typing import Dict, Any, List, Optional
import os

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain
from .guardrails import (
    GuardrailsState,
    create_guardrails_guard,
    create_factuality_guard,
    create_guardrails_node
)


class AgentState(TypedDict):
    """State schema for agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]
    helpfulness: Optional[str]  # Stores helpfulness decision: "Y", "N", or "END"


def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""
    
    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information from the student loan documents."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
    
    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent.
    
    Args:
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        List of tools
    """
    tools = []
    
    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))
    
    # Add Arxiv tool
    tools.append(ArxivQueryRun())
    
    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))
    
    return tools


def create_langgraph_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a simple LangGraph agent.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return END
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
    graph.add_edge("action", "agent")
    
    return graph.compile()


def route_to_action_or_helpfulness(state: AgentState):
    """Decide whether to execute tools or run the helpfulness evaluator."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "helpfulness"


def helpfulness_node(state: AgentState) -> Dict[str, Any]:
    """Evaluate helpfulness of the latest response relative to the initial query."""
    # If we've exceeded loop limit, add a message and set END marker
    if len(state["messages"]) > 10:
        unable_to_help_message = AIMessage(
            content="I apologize, but I'm unable to provide a helpful response to your query after multiple attempts. I may not have sufficient information to address this question adequately."
        )
        return {
            "messages": [unable_to_help_message],
            "helpfulness": "END"
        }
    
    initial_query = state["messages"][0]
    final_response = state["messages"][-1]
    
    prompt_template = """
Given an initial query and a final response, determine if the final response is extremely helpful or not. Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

Initial Query:
{initial_query}

Final Response:
{final_response}"""
    
    helpfulness_prompt_template = PromptTemplate.from_template(prompt_template)
    helpfulness_check_model = get_openai_model(model_name="gpt-4.1-mini")
    helpfulness_chain = (
        helpfulness_prompt_template | helpfulness_check_model | StrOutputParser()
    )
    
    helpfulness_response = helpfulness_chain.invoke(
        {
            "initial_query": initial_query.content,
            "final_response": final_response.content,
        }
    )
    
    decision = "Y" if "Y" in helpfulness_response else "N"
    return {"helpfulness": decision}


def helpfulness_decision(state: AgentState):
    """Terminate on 'Y' or loop otherwise; guard against infinite loops."""
    helpfulness = state.get("helpfulness")
    
    # Check loop-limit marker
    if helpfulness == "END":
        return END
    
    # If helpful, end; otherwise continue
    if helpfulness == "Y":
        return "end"
    return "continue"


def create_agent_with_helpfulness(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a LangGraph agent with helpfulness evaluation.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent with helpfulness checking
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Build graph with helpfulness evaluation
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("helpfulness", helpfulness_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_helpfulness,
        {"action": "action", "helpfulness": "helpfulness"},
    )
    graph.add_conditional_edges(
        "helpfulness",
        helpfulness_decision,
        {"continue": "agent", "end": END, END: END},
    )
    graph.add_edge("action", "agent")
    
    return graph.compile()


def create_input_validation_router(strict_mode: bool = True):
    """Create a router function for input validation."""
    def route_after_input_validation(state: GuardrailsState):
        """Route based on input validation results."""
        if not strict_mode:
            # In lenient mode, always continue to agent
            return "agent"
        validation_results = state.get("validation_results", [])
        if validation_results and not validation_results[-1].get("passed", True):
            return "error_handler"
        return "agent"
    return route_after_input_validation


def create_output_validation_router(strict_mode: bool = True):
    """Create a router function for output validation."""
    def route_after_output_validation(state: GuardrailsState):
        """Route based on output validation results."""
        if not strict_mode:
            # In lenient mode, always end successfully
            return END
        validation_results = state.get("validation_results", [])
        if validation_results and not validation_results[-1].get("passed", True):
            return "error_handler"
        return END
    return route_after_output_validation


def error_handler_node(state: GuardrailsState) -> Dict[str, Any]:
    """Handle validation errors and return appropriate error message to user."""
    validation_results = state.get("validation_results", [])
    
    # Get the last validation result
    if validation_results:
        last_result = validation_results[-1]
        error_type = last_result.get("type", "unknown")
        error_msg = last_result.get("error", "Validation failed")
        
        # Create appropriate error message based on validation type
        if error_type == "input":
            error_message = AIMessage(
                content="I apologize, but your query did not pass our validation checks. Please rephrase your question to be about student loans, financial aid, or education financing."
            )
        elif error_type == "output":
            error_message = AIMessage(
                content="I apologize, but I was unable to generate a response that meets our quality standards. Please try rephrasing your question."
            )
        else:
            error_message = AIMessage(
                content=f"I apologize, but an error occurred during validation: {error_msg}"
            )
    else:
        error_message = AIMessage(
            content="I apologize, but a validation error occurred. Please try again."
        )
    
    return {"messages": [error_message], "debug": state.get("debug", "")}


def create_agent_with_guardrails(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    input_guard_config: Optional[Dict[str, Any]] = None,
    output_guard_config: Optional[Dict[str, Any]] = None,
    strict_mode: bool = True
):
    """Create a LangGraph agent with Guardrails validation.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        input_guard_config: Optional dict with input guard configuration:
            - valid_topics: List of valid topics
            - invalid_topics: List of invalid topics
            - enable_jailbreak_detection: bool
            - enable_pii_protection: bool
            - enable_profanity_check: bool
            - pii_entities: List of PII entity types
        output_guard_config: Optional dict with output guard configuration:
            - eval_model: Model name for factuality evaluation
            - on_prompt: Whether to validate at prompt stage
        strict_mode: If True, routes to error_handler on validation failure.
            If False, logs warnings but continues. Default: True.
        
    Returns:
        Compiled LangGraph agent with guardrails validation
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: GuardrailsState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: GuardrailsState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "validate_output"
    
    # Create input guard with defaults
    if input_guard_config is None:
        input_guard_config = {
            "valid_topics": ["student loans", "financial aid", "education financing", "loan repayment"],
            "invalid_topics": ["investment advice", "crypto", "gambling", "politics"],
            "enable_jailbreak_detection": True,
            "enable_pii_protection": True,
            "enable_profanity_check": True,
            "enable_competitor_check": False
        }
    
    input_guard = create_guardrails_guard(**input_guard_config)
    
    # Create output guard with defaults
    if output_guard_config is None:
        output_guard_config = {
            "eval_model": "gpt-4.1-mini",
            "on_prompt": True
        }
    
    # For output, we'll use factuality guard
    output_guard = create_factuality_guard(**output_guard_config)
    
    # Create validation nodes
    # Note: We use strict_mode=False in nodes so failures are recorded in validation_results
    # and can be routed to error_handler. The strict_mode parameter controls whether
    # we enforce strict validation (route to error_handler) vs lenient (log and continue).
    input_validation_node = create_guardrails_node(
        input_guard=input_guard,
        output_guard=None,
        strict_mode=strict_mode  # Don't raise exceptions, record in validation_results
    )
    
    output_validation_node = create_guardrails_node(
        input_guard=None,
        output_guard=output_guard,
        strict_mode=strict_mode  # Don't raise exceptions, record in validation_results
    )
    
    # Build graph with guardrails
    graph = StateGraph(GuardrailsState)
    tool_node = ToolNode(tools)
    
    graph.add_node("validate_input", input_validation_node)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("validate_output", output_validation_node)
    graph.add_node("error_handler", error_handler_node)
    
    # Set entry point to input validation
    graph.set_entry_point("validate_input")
    
    # Create routing functions with strict_mode
    input_router = create_input_validation_router(strict_mode)
    output_router = create_output_validation_router(strict_mode)
    
    # Route after input validation
    graph.add_conditional_edges(
        "validate_input",
        input_router,
        {"agent": "agent", "error_handler": "error_handler"}
    )
    
    # Route after agent (to tools or output validation)
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"action": "action", "validate_output": "validate_output"}
    )
    
    # Route after output validation
    graph.add_conditional_edges(
        "validate_output",
        output_router,
        {"error_handler": "error_handler", END: END}
    )
    
    # Error handler always ends
    graph.add_edge("error_handler", END)
    
    # Action always goes back to agent
    graph.add_edge("action", "agent")
    
    return graph.compile()
