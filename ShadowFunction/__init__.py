import fastapi
from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import json
import os
import logging
import asyncio

from tools.searchshadow import SearchShadow
from tools.searchcustomer import SearchCustomer
from tools.searchuser import SearchUser

from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureOpenAISettings, OpenAISettings
from semantic_kernel.agents import OpenAIAssistantAgent, AssistantAgentThread
from semantic_kernel.contents.chat_message_content import (
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
)
from semantic_kernel.contents.utils.author_role import AuthorRole

# Import the modified plugin class
from plugins.shadow_insights_plugin import ShadowInsightsPlugin

from typing import Optional, AsyncGenerator
import asyncio
from asyncio import Queue

app = fastapi.FastAPI()

# Allow requests from all domains (not always recommended for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure module-level logger
logger = logging.getLogger("__init__.py")
logger.setLevel(logging.INFO)


# Define request body model
class ShadowRequest(BaseModel):
    query: str
    threadId: str
    additional_instructions: Optional[str] = None
    user_company: Optional[str] = None
    target_account: Optional[str] = None
    demand_stage: Optional[str] = None


# Instantiate search clients as singletons (if they are thread-safe or handle concurrency internally)
search_shadow_client = SearchShadow()
search_customer_client = SearchCustomer()
search_user_client = SearchUser()

ASSISTANT_ID = os.environ.get("ASSISTANT_ID")


async def get_agent() -> Optional[OpenAIAssistantAgent]:
    """
    Setup the Assistant with error handling.
    """
    try:
        # (2) Create plugin
        # Instantiate ShadowInsightsPlugin and pass the search clients
        shadow_plugin = ShadowInsightsPlugin(
            search_shadow_client, search_customer_client, search_user_client
        )
    except Exception as e:
        logger.error("Failed to instantiate ShadowInsightsPlugin: %s", e)
        return None

    try:
        # Create the client using Azure OpenAI resources and configuration
        client = OpenAIAssistantAgent.create_client(ai_model_id="gpt-4o")

        # Define the assistant definition
        definition = await client.beta.assistants.retrieve(
            ASSISTANT_ID
        )

        # Create the OpenAIAssistantAgent instance using the client and the assistant definition and the defined plugin
        agent = OpenAIAssistantAgent(
            client=client,
            definition=definition,
            plugins=[shadow_plugin],
        )

        if agent is None:
            logger.error(
                "Failed to retrieve the assistant agent. Please check the assistant ID."
            )
            return None
    except Exception as e:
        logger.error("An error occurred while retrieving the assistant agent: %s", e)
        return None

    return agent


async def event_stream(request: ShadowRequest) -> AsyncGenerator[str, None]:
    """
    Asynchronously stream responses back to the caller in JSON lines with granular events.
    
    This function now streams intermediate events (function calls, function results, etc.)
    to the client in real-time using Server-Sent Events (SSE).
    
    Event Types:
    - thread_info: Information about the agent and thread
    - function_call: When a tool/function is being called
    - function_result: When a tool/function returns a result
    - content: The actual response content from the agent
    - intermediate: Other intermediate processing events
    - stream_complete: Signals the end of the stream
    - error: Error events
    """
    agent = await get_agent()

    # Create a queue to pass intermediate events from the callback to the main stream
    event_queue: Queue = Queue()    # Modified callback function that yields events to the queue
    async def handle_streaming_intermediate_steps(message: ChatMessageContent) -> None:
        # Log that the callback was triggered
        print(f"Intermediate callback triggered with message role: {message.role}")
        
        for item in message.items or []:
            try:
                if isinstance(item, FunctionCallContent):
                    # Safely serialize the arguments
                    arguments = item.arguments
                    if arguments and not isinstance(arguments, (str, dict)):
                        arguments = str(arguments)
                    
                    event_data = {
                        "type": "function_call",
                        "function_name": item.name,
                        "arguments": arguments
                    }
                    await event_queue.put(("function_call", event_data))
                    print(f"[FUNC CALL] {item.name} with arguments: {arguments}")
                    
                elif isinstance(item, FunctionResultContent):
                    # Safely serialize the result
                    result = item.result
                    if not isinstance(result, (str, int, float, bool, list, dict, type(None))):
                        result = str(result)
                    
                    event_data = {
                        "type": "function_result",
                        "function_name": item.name,
                        "result": result
                    }
                    await event_queue.put(("function_result", event_data))
                    print(f"[FUNC RESULT] {result} for function: {item.name}")
                else:
                    # For other types, convert to string safely
                    content = str(item)
                    event_data = {
                        "type": "intermediate",
                        "content": content
                    }
                    await event_queue.put(("intermediate", event_data))
                    print(f"[INTERMEDIATE] {content}")
            except Exception as e:
                # If there's an error processing an individual item, log it but continue
                print(f"Error processing intermediate item: {e}")
                error_event = {
                    "type": "intermediate_error",
                    "error": str(e),
                    "item_type": type(item).__name__
                }
                await event_queue.put(("intermediate_error", error_event))

    # Extract fields directly
    query = request.query  # required field, always present
    threadId = request.threadId  # required field, always present
    user_company = request.user_company  # optional
    target_account = request.target_account  # optional
    demand_stage = request.demand_stage  # optional

    # Build structured parameters
    params = {
        "target_account": target_account,
        "user_company": user_company,
        "demand_stage": demand_stage,
    }

    # Combine query and parameters into a single string
    combined_query = f"{query} - {params}"

    # Retrieve or create a thread ID
    if threadId:
        # Create the AssistantAgentThread instance with the existing thread_id
        current_thread = AssistantAgentThread(client=agent.client, thread_id=threadId)
    else:
        # Create a new thread because none was passed
        current_thread: AssistantAgentThread = None

    # Get any additional instructions passed for the assistant
    additional_instructions = (
        f"<additional_instructions>{request.additional_instructions}</additional_instructions>"
        if request.additional_instructions
        else None
    )

    # Task to consume events from the queue and stream them to the client
    async def stream_events():
        try:
            first_chunk = True
            stream_finished = False
              # Start the agent streaming in a background task
            async def agent_stream_task():
                nonlocal stream_finished, first_chunk
                try:
                    async for response in agent.invoke_stream(
                        messages=combined_query,
                        thread=current_thread,
                        additional_instructions=additional_instructions,
                        on_intermediate_message=handle_streaming_intermediate_steps,
                    ):
                        if first_chunk:
                            thread_info = {
                                "type": "thread_info",
                                "agent_name": getattr(response, 'name', 'Unknown'),
                                "thread_id": getattr(response.thread, 'id', 'Unknown') if hasattr(response, 'thread') else 'Unknown'
                            }
                            await event_queue.put(("thread_info", thread_info))
                            print(f"# {getattr(response, 'name', 'Agent')}: ", end="", flush=True)
                            if hasattr(response, 'thread') and hasattr(response.thread, 'id'):
                                print(f"Using Thread ID: {response.thread.id}")
                            first_chunk = False
                          # Check if this response contains function calls or results in its items
                        if hasattr(response, 'items') and response.items:
                            for item in response.items:
                                if isinstance(item, FunctionCallContent):
                                    # Safely serialize the arguments
                                    arguments = item.arguments
                                    if arguments and not isinstance(arguments, (str, dict)):
                                        arguments = str(arguments)
                                    
                                    event_data = {
                                        "type": "function_call",
                                        "function_name": item.name,
                                        "arguments": arguments
                                    }
                                    await event_queue.put(("function_call", event_data))
                                    print(f"[MAIN FUNC CALL] {item.name} with arguments: {arguments}")
                                    
                                elif isinstance(item, FunctionResultContent):
                                    # Safely serialize the result
                                    result = item.result
                                    if not isinstance(result, (str, int, float, bool, list, dict, type(None))):
                                        result = str(result)
                                    
                                    event_data = {
                                        "type": "function_result",
                                        "function_name": item.name,
                                        "result": result
                                    }
                                    await event_queue.put(("function_result", event_data))
                                    print(f"[MAIN FUNC RESULT] {result} for function: {item.name}")
                        
                        # Extract content safely from the streaming response
                        content = ""
                        if hasattr(response, 'content') and response.content is not None:
                            content = str(response.content)
                        
                        # Only send content events if there's actual content
                        if content.strip():
                            content_data = {
                                "type": "content",
                                "content": content
                            }
                            await event_queue.put(("content", content_data))
                            print(content, end="", flush=True)
                    
                    print()
                    # Signal that streaming is complete
                    await event_queue.put(("stream_complete", {"type": "stream_complete"}))
                except Exception as e:
                    error_data = {"type": "error", "error": str(e)}
                    await event_queue.put(("error", error_data))
                finally:
                    stream_finished = True

            # Start the agent streaming task
            agent_task = asyncio.create_task(agent_stream_task())

            # Stream events to the client
            while not stream_finished or not event_queue.empty():
                try:
                    # Wait for an event with a timeout to check if streaming is done
                    event_type, event_data = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    
                    # Yield the event as SSE format
                    yield f"event: {event_type}\n"
                    yield f"data: {json.dumps(event_data)}\n\n"
                    
                    if event_type == "stream_complete":
                        break
                        
                except asyncio.TimeoutError:
                    # Continue the loop to check if streaming is finished
                    continue
                except Exception as e:
                    error_data = json.dumps({"type": "error", "error": str(e)})
                    yield "event: error\n"
                    yield f"data: {error_data}\n\n"
                    break

            # Wait for the agent task to complete
            await agent_task

        except HTTPException as exc:
            # SSE error payload
            error_data = json.dumps({"error": exc.detail})
            yield "event: error\n"
            yield f"data: {error_data}\n\n"
        except Exception as e:
            logging.exception("Unexpected error during streaming SSE.")
            error_data = json.dumps({"error": str(e)})
            yield "event: error\n"
            yield f"data: {error_data}\n\n"

    # Stream all events to the client
    async for event in stream_events():
        yield event


@app.post("/shadow-sk")
async def shadow_sk(request: ShadowRequest):
    """
    Endpoint that receives a query, passes it to the agent, and streams back responses.
    """
    return StreamingResponse(
        event_stream(request),
        media_type="text/plain",
        status_code=200,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )
