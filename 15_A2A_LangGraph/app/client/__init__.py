"""Client package for A2A protocol interaction.

This package contains the client-side implementation for communicating
with A2A-compliant agent servers.
"""
from app.client.a2a_tool import query_a2a_agent
from app.client.client_agent import ClientAgent

__all__ = ["query_a2a_agent", "ClientAgent"]

