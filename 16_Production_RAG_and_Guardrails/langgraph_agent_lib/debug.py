"""Debug utilities for LangGraph agent execution.

This module provides debugging tools to visualize and trace agent execution,
including node transitions, message flow, and state changes.
"""

from typing import Dict, Any, Optional, List
import time
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage


def invoke_with_debug(
    agent,
    input_state: Dict[str, Any],
    verbose: bool = True,
    show_messages: bool = True,
    show_validation: bool = True,
    show_timing: bool = True
) -> Dict[str, Any]:
    """Invoke agent with debug output showing each execution step.
    
    Args:
        agent: The compiled LangGraph agent to execute.
        input_state: Initial state dictionary (e.g., {"messages": [...]}).
        verbose: If True, show detailed output for each step. Default: True.
        show_messages: If True, show message details. Default: True.
        show_validation: If True, show validation results. Default: True.
        show_timing: If True, show timing information. Default: True.
        
    Returns:
        Final state dictionary after execution.
    """
    print("=" * 70)
    print("🔄 STARTING AGENT EXECUTION (Debug Mode)")
    print("=" * 70)
    
    current_state = input_state.copy()
    step = 0
    start_time = time.time()
    node_timings = []
    
    # Show initial state
    if verbose:
        print(f"\n📥 Initial State:")
        if "messages" in current_state:
            print(f"   Messages: {len(current_state['messages'])}")
            if current_state['messages']:
                initial_msg = current_state['messages'][0]
                if isinstance(initial_msg, HumanMessage):
                    preview = initial_msg.content[:80] if hasattr(initial_msg, 'content') else str(initial_msg)[:80]
                    print(f"   Query: {preview}...")
        print()
    
    # Stream execution
    for state_update in agent.stream(input_state):
        step += 1
        node_start_time = time.time()
        
        # Get node name and state
        node_name = list(state_update.keys())[0] if state_update else "unknown"
        node_state = state_update.get(node_name, {})
        
        if verbose:
            print(f"{'─' * 70}")
            print(f"Step {step}: Node '{node_name}'")
            print(f"{'─' * 70}")
        
        # Show messages
        if show_messages and "messages" in node_state:
            messages = node_state["messages"]
            if isinstance(messages, list):
                # Find new messages (those not in previous state)
                prev_message_count = len(current_state.get("messages", []))
                new_messages = messages[prev_message_count:] if prev_message_count < len(messages) else []
                if not new_messages:
                    # If no new messages detected, show all (might be state update)
                    new_messages = messages
                    
                for msg in new_messages:
                    msg_type = type(msg).__name__
                    if isinstance(msg, HumanMessage):
                        content_preview = msg.content[:100] if hasattr(msg, 'content') else str(msg)[:100]
                        print(f"   📥 {msg_type}: {content_preview}...")
                    elif isinstance(msg, AIMessage):
                        content_preview = msg.content[:100] if hasattr(msg, 'content') else str(msg)[:100]
                        tool_calls = getattr(msg, 'tool_calls', None)
                        if tool_calls:
                            print(f"   📤 {msg_type}: {content_preview}...")
                            print(f"      Tool calls: {len(tool_calls)}")
                            for tc in tool_calls[:3]:  # Show first 3 tool calls
                                tool_name = getattr(tc, 'name', 'unknown')
                                print(f"        - {tool_name}")
                            if len(tool_calls) > 3:
                                print(f"        ... and {len(tool_calls) - 3} more")
                        else:
                            print(f"   📤 {msg_type}: {content_preview}...")
                    elif isinstance(msg, ToolMessage):
                        tool_name = getattr(msg, 'name', 'unknown_tool')
                        content_preview = msg.content[:150] if hasattr(msg, 'content') else str(msg)[:150]
                        print(f"   🔧 ToolMessage [{tool_name}]: {content_preview}...")
                print(f"   Total messages: {len(messages)}")
        
        # Show validation results
        if show_validation and "validation_results" in node_state:
            validation = node_state["validation_results"]
            if validation:
                print(f"   🔍 Validation Results:")
                for v_result in validation:
                    v_type = v_result.get("type", "unknown")
                    passed = v_result.get("passed", False)
                    status = "✅ PASSED" if passed else "❌ FAILED"
                    error = v_result.get("error")
                    print(f"      {status} - {v_type}")
                    if error:
                        print(f"        Error: {error[:100]}...")
        
        # Show other state updates
        other_keys = [k for k in node_state.keys() if k not in ["messages", "validation_results"]]
        if other_keys and verbose:
            for key in other_keys:
                value = node_state[key]
                if isinstance(value, str) and len(value) > 100:
                    print(f"   {key}: {value[:100]}...")
                else:
                    print(f"   {key}: {value}")
        
        # Update current state
        current_state = {**current_state, **state_update}
        
        # Timing
        if show_timing:
            node_elapsed = time.time() - node_start_time
            node_timings.append((node_name, node_elapsed))
            print(f"   ⏱️  Node execution time: {node_elapsed:.3f}s")
        
        if verbose:
            print()
    
    # Final summary
    total_time = time.time() - start_time
    print("=" * 70)
    print("✅ EXECUTION COMPLETE")
    print("=" * 70)
    
    if show_timing:
        print(f"\n⏱️  Timing Summary:")
        print(f"   Total execution time: {total_time:.3f}s")
        if node_timings:
            print(f"   Node timings:")
            for node_name, elapsed in node_timings:
                percentage = (elapsed / total_time * 100) if total_time > 0 else 0
                print(f"      {node_name}: {elapsed:.3f}s ({percentage:.1f}%)")
    
    if verbose:
        print(f"\n📊 Final State:")
        if "messages" in current_state:
            print(f"   Total messages: {len(current_state['messages'])}")
            final_msg = current_state['messages'][-1] if current_state['messages'] else None
            if final_msg and isinstance(final_msg, AIMessage):
                print(f"   Final response: {final_msg.content[:150] if hasattr(final_msg, 'content') else str(final_msg)[:150]}...")
        if "validation_results" in current_state:
            validation_count = len(current_state['validation_results'])
            print(f"   Validation checks: {validation_count}")
    
    print()
    
    return current_state


def stream_events_with_debug(
    agent,
    input_state: Dict[str, Any],
    show_llm_calls: bool = True,
    show_tool_calls: bool = True,
    show_node_transitions: bool = True
) -> Dict[str, Any]:
    """Stream agent events with detailed debug output.
    
    Args:
        agent: The compiled LangGraph agent to execute.
        input_state: Initial state dictionary.
        show_llm_calls: If True, show LLM API calls. Default: True.
        show_tool_calls: If True, show tool invocations. Default: True.
        show_node_transitions: If True, show node transitions. Default: True.
        
    Returns:
        Final state dictionary after execution.
    """
    print("=" * 70)
    print("🔄 STREAMING EVENTS (Detailed Debug Mode)")
    print("=" * 70)
    print()
    
    events = []
    current_state = input_state.copy()
    
    for event in agent.stream_events(input_state, version="v2"):
        events.append(event)
        kind = event.get("event")
        name = event.get("name", "unknown")
        
        if kind == "on_chain_start" and show_node_transitions:
            print(f"▶️  Starting: {name}")
            if "data" in event:
                print(f"   Input: {str(event['data'])[:100]}...")
                
        elif kind == "on_chain_end" and show_node_transitions:
            print(f"✅ Completed: {name}")
            if "data" in event:
                output = event["data"].get("output", {})
                if "messages" in output:
                    msg_count = len(output["messages"])
                    print(f"   Messages: {msg_count}")
                    
        elif kind == "on_chain_error" and show_node_transitions:
            print(f"❌ Error in: {name}")
            if "error" in event:
                print(f"   Error: {str(event['error'])[:200]}")
                
        elif kind == "on_chat_model_start" and show_llm_calls:
            print(f"🤖 LLM API Call Starting...")
            if "data" in event and "messages" in event["data"]:
                msg_count = len(event["data"]["messages"])
                print(f"   Messages sent: {msg_count}")
                
        elif kind == "on_chat_model_end" and show_llm_calls:
            print(f"🤖 LLM API Call Completed")
            if "data" in event and "output" in event["data"]:
                output = event["data"]["output"]
                if hasattr(output, 'content'):
                    preview = output.content[:80]
                    print(f"   Response: {preview}...")
                    
        elif kind == "on_tool_start" and show_tool_calls:
            tool_name = event.get("name", "unknown_tool")
            print(f"🔧 Tool Call Starting: {tool_name}")
            if "data" in event and "input" in event["data"]:
                tool_input = str(event["data"]["input"])[:100]
                print(f"   Input: {tool_input}...")
                
        elif kind == "on_tool_end" and show_tool_calls:
            tool_name = event.get("name", "unknown_tool")
            print(f"🔧 Tool Call Completed: {tool_name}")
            if "data" in event and "output" in event["data"]:
                tool_output = str(event["data"]["output"])[:100]
                print(f"   Output: {tool_output}...")
    
    # Collect final state from events
    for event in reversed(events):
        if event.get("event") == "on_chain_end" and "data" in event:
            if "output" in event["data"]:
                current_state = {**current_state, **event["data"]["output"]}
                break
    
    print()
    print("=" * 70)
    print("✅ EVENT STREAMING COMPLETE")
    print("=" * 70)
    print(f"Total events captured: {len(events)}")
    print()
    
    return current_state


def print_graph_structure(agent):
    """Print the agent's graph structure in ASCII format.
    
    Args:
        agent: The compiled LangGraph agent.
    """
    print("=" * 70)
    print("📊 AGENT GRAPH STRUCTURE")
    print("=" * 70)
    print()
    try:
        agent.get_graph().print_ascii()
    except Exception as e:
        print(f"Could not print graph structure: {e}")
        print("Trying alternative method...")
        try:
            print(str(agent.get_graph()))
        except:
            print("Graph structure not available")
    print()

