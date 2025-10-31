"""Client Agent that uses A2A protocol to interact with the A2A server.

This module implements a LangGraph-based agent that can intelligently decide
when and how to use an A2A-compliant agent server as a tool.
"""
import os
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.client.a2a_tool import query_a2a_agent


class ClientAgent:
    """A LangGraph agent that can query an A2A server as a tool.
    
    This agent acts as an intelligent wrapper around the A2A protocol,
    deciding when to leverage the A2A server's capabilities (web search,
    academic paper search, document retrieval) based on user queries.
    """
    
    SYSTEM_PROMPT = """You are an intelligent assistant that has access to a powerful A2A agent server.

The A2A agent server has the following capabilities:
1. Web Search - Can search the internet for current information and news
2. Academic Paper Search - Can search arXiv for research papers
3. Document Retrieval (RAG) - Can search through loaded documents

When to use the A2A agent tool:
- When users ask about current events, news, or recent developments
- When users want to find academic papers or research
- When users ask questions that require searching through documents
- When you need factual, up-to-date information that you might not have

How to use the tool effectively:
- Be specific and clear in your queries to the A2A agent
- Pass the user's question directly or rephrase it for clarity
- Interpret and present the A2A agent's responses in a helpful way

Important:
- Always use the query_a2a_agent tool for searches and lookups
- Provide context and explanation when presenting results to users
- If the A2A agent returns an error, explain it to the user and suggest alternatives
"""
    
    def __init__(self):
        """Initialize the client agent with LLM and A2A tool."""
        self.model = ChatOpenAI(
            model=os.getenv('TOOL_LLM_NAME', 'gpt-4o-mini'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_api_base=os.getenv('TOOL_LLM_URL', 'https://api.openai.com/v1'),
            temperature=0,
        )
        
        # Create agent with the A2A tool
        self.memory = MemorySaver()
        self.agent = create_react_agent(
            self.model,
            tools=[query_a2a_agent],
            checkpointer=self.memory
        )
    
    async def query(self, user_query: str, thread_id: str = "default") -> str:
        """Send a query to the client agent.
        
        Args:
            user_query: The user's question or request
            thread_id: Thread ID for conversation context
            
        Returns:
            str: The agent's response
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # Include system prompt with the first message
        messages = [("system", self.SYSTEM_PROMPT), ("user", user_query)]
        
        response_parts = []
        async for event in self.agent.astream(
            {"messages": messages},
            config=config
        ):
            # Extract the last message from each event
            for value in event.values():
                if "messages" in value:
                    last_message = value["messages"][-1]
                    # Check if it's the final AI message
                    if hasattr(last_message, "content") and last_message.content:
                        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                            response_parts.append(last_message.content)
        
        # Return the last response (the final answer)
        return response_parts[-1] if response_parts else "No response generated."
    
    def query_sync(self, user_query: str, thread_id: str = "default") -> str:
        """Synchronous version of query method.
        
        Args:
            user_query: The user's question or request
            thread_id: Thread ID for conversation context
            
        Returns:
            str: The agent's response
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # Include system prompt with the first message
        messages = [("system", self.SYSTEM_PROMPT), ("user", user_query)]
        
        result = self.agent.invoke(
            {"messages": messages},
            config=config
        )
        
        # Extract the final message content
        if "messages" in result:
            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                return last_message.content
        
        return "No response generated."

