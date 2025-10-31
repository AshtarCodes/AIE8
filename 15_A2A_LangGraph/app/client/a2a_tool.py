"""A2A Tool Wrapper for LangChain.

This module provides a LangChain tool that wraps the A2A client functionality,
allowing agents to communicate with A2A-compliant servers.
"""
import logging
from uuid import uuid4

import httpx
from langchain_core.tools import tool

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)


logger = logging.getLogger(__name__)


@tool
async def query_a2a_agent(query: str) -> str:
    """Query the A2A agent server with a question.
    
    This tool connects to an A2A-compliant agent server and sends queries to it.
    The server has access to web search, academic paper search (arXiv), and document
    retrieval (RAG) capabilities.
    
    Use this tool when you need to:
    - Search for current information on the web
    - Find academic papers and research
    - Retrieve information from documents
    
    Args:
        query: The question or request to send to the A2A agent
        
    Returns:
        str: The response from the A2A agent
    """
    base_url = 'http://localhost:10000'
    
    try:
        # Increase timeout for LLM responses (default is 5 seconds, which is too short)
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as httpx_client:
            # Initialize A2ACardResolver
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=base_url,
            )
            
            # Fetch Public Agent Card
            final_agent_card_to_use = None
            
            try:
                logger.info(
                    f'Attempting to fetch public agent card from: {base_url}{AGENT_CARD_WELL_KNOWN_PATH}'
                )
                _public_card = await resolver.get_agent_card()
                logger.info('Successfully fetched public agent card')
                final_agent_card_to_use = _public_card
                
                # Try to fetch extended card if supported
                if _public_card.supports_authenticated_extended_card:
                    try:
                        logger.info('Attempting to fetch authenticated extended card')
                        auth_headers_dict = {
                            'Authorization': 'Bearer dummy-token-for-extended-card'
                        }
                        _extended_card = await resolver.get_agent_card(
                            relative_card_path=EXTENDED_AGENT_CARD_PATH,
                            http_kwargs={'headers': auth_headers_dict},
                        )
                        logger.info('Successfully fetched extended agent card')
                        final_agent_card_to_use = _extended_card
                    except Exception as e_extended:
                        logger.warning(
                            f'Failed to fetch extended agent card: {e_extended}. '
                            'Will proceed with public card.'
                        )
                        
            except Exception as e:
                logger.error(f'Failed to fetch agent card: {e}')
                return f"Error: Unable to connect to A2A agent server at {base_url}. Is the server running?"
            
            # Initialize A2AClient and send message
            client = A2AClient(
                httpx_client=httpx_client, agent_card=final_agent_card_to_use
            )
            logger.info('A2AClient initialized.')
            
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': query}
                    ],
                    'message_id': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_message_payload)
            )
            
            response = await client.send_message(request)
            
            # Extract the message content from the response
            if response.root and response.root.result:
                result = response.root.result
                
                # Check for artifacts (completed tasks)
                if hasattr(result, 'artifacts') and result.artifacts:
                    response_texts = []
                    for artifact in result.artifacts:
                        if hasattr(artifact, 'parts') and artifact.parts:
                            for part in artifact.parts:
                                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                    response_texts.append(part.root.text)
                    if response_texts:
                        return " ".join(response_texts)
                
                # Check for message (some responses might have direct messages)
                if hasattr(result, 'message') and result.message and hasattr(result.message, 'parts'):
                    response_texts = []
                    for part in result.message.parts:
                        if hasattr(part, 'text') and part.text:
                            response_texts.append(part.text)
                    if response_texts:
                        return " ".join(response_texts)
                    
            return "Error: Received an empty or malformed response from the A2A agent."
            
    except httpx.TimeoutException:
        return "Error: Request to A2A agent timed out. The agent may be processing a complex query."
    except Exception as e:
        logger.error(f'Error querying A2A agent: {e}', exc_info=True)
        return f"Error: An unexpected error occurred while querying the A2A agent: {str(e)}"

