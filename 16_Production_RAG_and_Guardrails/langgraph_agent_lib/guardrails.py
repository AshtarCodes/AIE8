"""Guardrails integration for production-safe LangGraph agents.

This module provides utilities for integrating Guardrails AI validation
into LangGraph agent workflows, including input and output validation.
"""

import logging
from typing import Dict, Any, Optional, List
from typing_extensions import TypedDict, Annotated

from guardrails.hub import (
    RestrictToTopic,
    DetectJailbreak,
    CompetitorCheck,
    LlmRagEvaluator,
    HallucinationPrompt,
    ProfanityFree,
    GuardrailsPII
)
from guardrails import Guard
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GuardrailsState(TypedDict):
    """State schema for guardrails-enabled agent graphs.
    
    Attributes:
        messages: List of messages in the conversation history.
        validation_results: Optional validation results from guardrails.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    validation_results: Optional[Dict[str, Any]]


def create_guardrails_guard(
    valid_topics: Optional[List[str]] = None,
    invalid_topics: Optional[List[str]] = None,
    enable_jailbreak_detection: bool = True,
    enable_pii_protection: bool = True,
    enable_profanity_check: bool = True,
    enable_competitor_check: bool = False,
    pii_entities: Optional[List[str]] = None
) -> Guard:
    """Create a Guardrails guard with common production safety checks.
    
    Args:
        valid_topics: List of valid topics to allow. None disables topic restriction.
        invalid_topics: List of invalid topics to block. None disables topic restriction.
        enable_jailbreak_detection: Whether to enable jailbreak detection. Default: True.
        enable_pii_protection: Whether to enable PII detection and redaction. Default: True.
        enable_profanity_check: Whether to enable profanity filtering. Default: True.
        enable_competitor_check: Whether to enable competitor mention detection. Default: False.
        pii_entities: List of PII entity types to detect. Default: Common PII types.
        
    Returns:
        Configured Guard instance.
        
    Raises:
        RuntimeError: If guard configuration fails.
    """
    guard = Guard()
    
    try:
        # Topic restriction
        if valid_topics or invalid_topics:
            guard = guard.use(
                RestrictToTopic(
                    valid_topics=valid_topics or [],
                    invalid_topics=invalid_topics or [],
                    disable_classifier=True,
                    disable_llm=False,
                    on_fail="exception"
                )
            )
            logger.debug("Topic restriction guard configured")
        
        # Jailbreak detection
        if enable_jailbreak_detection:
            guard = guard.use(DetectJailbreak())
            logger.debug("Jailbreak detection guard configured")
        
        # PII protection
        if enable_pii_protection:
            default_entities = ["CREDIT_CARD", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"]
            entities = pii_entities or default_entities
            guard = guard.use(
                GuardrailsPII(
                    entities=entities,
                    on_fail="fix"
                )
            )
            logger.debug(f"PII protection guard configured for entities: {entities}")
        
        # Profanity check
        if enable_profanity_check:
            guard = guard.use(
                ProfanityFree(
                    threshold=0.8,
                    validation_method="sentence",
                    on_fail="exception"
                )
            )
            logger.debug("Profanity check guard configured")
        
        # Competitor check (optional)
        if enable_competitor_check:
            guard = guard.use(CompetitorCheck())
            logger.debug("Competitor check guard configured")
        
        logger.info("Guardrails guard configured successfully")
        return guard
        
    except Exception as e:
        logger.error(f"Failed to configure guardrails: {e}", exc_info=True)
        raise RuntimeError(f"Failed to configure guardrails: {e}") from e


def create_factuality_guard(
    eval_model: str = "gpt-4.1-mini",
    on_prompt: bool = True
) -> Guard:
    """Create a factuality guard for RAG responses.
    
    Args:
        eval_model: Model to use for factuality evaluation. Default: "gpt-4.1-mini".
        on_prompt: Whether to validate at prompt stage or response stage. Default: True.
        
    Returns:
        Configured Guard instance for factuality checking.
        
    Raises:
        RuntimeError: If guard configuration fails.
    """
    try:
        guard = Guard().use(
            LlmRagEvaluator(
                eval_llm_prompt_generator=HallucinationPrompt(prompt_name="hallucination_judge_llm"),
                llm_evaluator_fail_response="hallucinated",
                llm_evaluator_pass_response="factual",
                llm_callable=eval_model,
                on_fail="exception",
                on="prompt" if on_prompt else "response"
            )
        )
        logger.info(f"Factuality guard configured with model: {eval_model}")
        return guard
    except Exception as e:
        logger.error(f"Failed to configure factuality guard: {e}", exc_info=True)
        raise RuntimeError(f"Failed to configure factuality guard: {e}") from e


def validate_input(
    guard: Guard,
    user_input: str,
    raise_on_failure: bool = True
) -> Dict[str, Any]:
    """Validate user input using a Guardrails guard.
    
    Args:
        guard: The Guard instance to use for validation.
        user_input: The user input to validate.
        raise_on_failure: Whether to raise an exception on validation failure.
            If False, returns validation result. Default: True.
        
    Returns:
        Dictionary with validation results including:
        - validation_passed: Boolean indicating if validation passed
        - validated_output: The validated (and potentially modified) output
        - error: Error message if validation failed
        
    Raises:
        RuntimeError: If validation fails and raise_on_failure is True.
    """
    try:
        result = guard.validate(user_input)
        
        validation_result = {
            "validation_passed": result.validation_passed,
            "validated_output": result.validated_output if hasattr(result, 'validated_output') else user_input,
            "error": None
        }
        
        if not result.validation_passed and raise_on_failure:
            error_msg = f"Input validation failed: {getattr(result, 'error', 'Unknown error')}"
            logger.warning(f"Input validation failed: {user_input[:100]}...")
            raise RuntimeError(error_msg)
        
        return validation_result
        
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Input validation error: {e}", exc_info=True)
        if raise_on_failure:
            raise RuntimeError(f"Input validation failed: {e}") from e
        return {
            "validation_passed": False,
            "validated_output": user_input,
            "error": str(e)
        }


def validate_output(
    guard: Guard,
    agent_response: str,
    original_prompt: str,
    context: Optional[str] = None,
    raise_on_failure: bool = True
) -> Dict[str, Any]:
    """Validate agent output using a Guardrails guard.
    
    Args:
        guard: The Guard instance to use for validation.
        agent_response: The agent's response to validate.
        original_prompt: Optional original prompt that generated this response.
            Required by LlmRagEvaluator for factuality checking.
        context: Optional additional context for factuality checking (e.g., retrieved documents).
            Can be provided in addition to original_prompt.
        raise_on_failure: Whether to raise an exception on validation failure.
            If False, returns validation result. Default: True.
        
    Returns:
        Dictionary with validation results.
        
    Raises:
        RuntimeError: If validation fails and raise_on_failure is True.
    """
    try:
        logger.debug(f"Validating output with guardrails: {agent_response}")
        logger.debug(f"Original prompt: {original_prompt}")
        logger.debug(f"Context: {context}")
        metadata = { "user_message": original_prompt, "context": context, "llm_response": agent_response, "original_prompt": original_prompt, "reference_text": context }
        # result = guard.parse(agent_response, metadata=metadata)
        result = guard.validate(llm_output=agent_response, metadata=metadata)
        
        validation_result = {
            "validation_passed": result.validation_passed,
            "validated_output": result.validated_output if hasattr(result, 'validated_output') else agent_response,
            "error": None,
            "debug": { "original_prompt": original_prompt, "context": context, "llm_response": agent_response }
        }
        
        if not result.validation_passed and raise_on_failure:
            error_msg = f"Output validation failed: {getattr(result, 'error', 'Unknown error')}"
            logger.warning(f"Output validation failed: {agent_response[:100]}...")
            raise RuntimeError(error_msg)
        
        return validation_result
        
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Output validation error: {e}", exc_info=True)
        if raise_on_failure:
            raise RuntimeError(f"Output validation failed: {e}") from e
        return {
            "validation_passed": False,
            "validated_output": agent_response,
            "error": str(e)
        }

def get_context(messages: List[BaseMessage]) -> str:
    """Get context from messages for factuality checking."""
    context_text = ""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) or msg.type == "tool":
            context_text += msg.content
    if not context_text:
        for msg in messages:
            if isinstance(msg, HumanMessage):
                context_text = msg.content
                break

    return context_text


def create_guardrails_node(
    input_guard: Optional[Guard] = None,
    output_guard: Optional[Guard] = None,
    strict_mode: bool = True
):
    """Create a LangGraph node that validates inputs and outputs with Guardrails.
    
    Args:
        input_guard: Guard for validating user inputs. If None, input validation is skipped.
        output_guard: Guard for validating agent outputs. If None, output validation is skipped.
        strict_mode: If True, raises exceptions on validation failure.
            If False, logs warnings but continues. Default: True.
        
    Returns:
        A function that can be used as a LangGraph node.
    """
    def guardrails_node(state: GuardrailsState) -> Dict[str, Any]:
        """Validate messages in the agent state.
        
        Args:
            state: Current agent state with messages.
            
        Returns:
            Updated state with validation results.
        """
        messages = state["messages"]
        validation_results = []
        
        if not messages:
            return {"validation_results": []}
        
        # Validate the last message
        last_message = messages[-1]
        
        try:
            if isinstance(last_message, HumanMessage) and input_guard:
                result = validate_input(
                    input_guard,
                    last_message.content,
                    raise_on_failure=strict_mode
                )

                validation_results.append({
                    "type": "input",
                    "passed": result["validation_passed"],
                    "message": last_message.content
                })
                
                # If validation modified the input, we could update the message here
                if not result["validation_passed"] and strict_mode:
                    logger.error(f"Input validation failed: {result.get('error')}")
            
            elif isinstance(last_message, AIMessage) and output_guard:
                original_prompt = messages[0].content
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        original_prompt = msg.content
                        break
                
                # Extract context from tool outputs (ToolMessage objects)
                # These contain the results from RAG, search, and other tools
                context_text = get_context(messages)
                # Pass original_prompt and context for factuality guards
                result = validate_output(
                    guard=output_guard,
                    agent_response=last_message.content,
                    original_prompt=original_prompt,
                    context=context_text,
                    raise_on_failure=strict_mode
                )
                validation_results.append({
                    "type": "output",
                    "passed": result["validation_passed"],
                    "message": last_message.content,
                    "debug": { "original_prompt": original_prompt, "context": context_text, "llm_response": last_message.content }
                })
                
                if not result["validation_passed"] and strict_mode:
                    logger.error(f"Output validation failed: {result.get('error')}")
                    
        except Exception as e:
            logger.error(f"Guardrails validation error: {e}", exc_info=True)
            if strict_mode:
                raise e
            validation_results.append({
                "type": "error",
                "passed": False,
                "error": str(e),
                "debug": { "original_prompt": original_prompt, "context": context_text, "llm_response": last_message.content }
            })
        
        return {"validation_results": validation_results, "debug": result.get("debug", "")}
    
    return guardrails_node

