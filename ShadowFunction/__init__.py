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
from semantic_kernel.agents.open_ai import OpenAIAssistantAgent
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

# Import the modified plugin class
from plugins.shadow_insights_plugin import ShadowInsightsPlugin

from typing import Optional

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
        # (1) Create the instance of the Kernel
        kernel = Kernel()
    except Exception as e:
        logger.error("Failed to initialize the kernel: %s", e)
        return None

    try:
        # (2) Add plugin
        # Instantiate ShadowInsightsPlugin and pass the search clients
        shadow_plugin = ShadowInsightsPlugin(search_shadow_client, search_customer_client, search_user_client)
    except Exception as e:
        logger.error("Failed to instantiate ShadowInsightsPlugin: %s", e)
        return None

    try:
        # (3) Register plugin with the Kernel
        kernel.add_plugin(shadow_plugin, plugin_name="shadowRetrievalPlugin")
    except Exception as e:
        logger.error("Failed to register plugin with the kernel: %s", e)
        return None

    try:
        # (4) Retrieve the agent
        agent = await OpenAIAssistantAgent.retrieve(
            id=ASSISTANT_ID, kernel=kernel, ai_model_id="gpt-4o"
        )
        if agent is None:
            logger.error("Failed to retrieve the assistant agent. Please check the assistant ID.")
            return None
    except Exception as e:
        logger.error("An error occurred while retrieving the assistant agent: %s", e)
        return None

    return agent

@app.post("/shadow-sk")
async def shadow_sk(request: ShadowRequest):
    """
    Endpoint that receives a query, passes it to the agent, and streams back responses.
    """
    agent = await get_agent()

    # Extract fields directly
    query = request.query  # required field, always present
    threadId = request.threadId  # required field, always present
    user_company = request.user_company  # optional
    target_account = request.target_account  # optional
    demand_stage = request.demand_stage # optional

    # Build structured parameters
    params = {
        "target_account": target_account,
        "user_company": user_company,
        "demand_stage": demand_stage
    }    # Combine query and parameters into a single string
    combined_query = f"{query} - {params}"

    # Retrieve or create a thread ID
    if threadId:
        current_thread_id = threadId
    else:
        current_thread_id = await agent.create_thread()  # create new threadId and set as current

    # Create the user message content with the request.query
    message_user = ChatMessageContent(role=AuthorRole.USER, content=combined_query)
    # Add the user message to the agent
    await agent.add_chat_message(thread_id=current_thread_id, message=message_user)
      # get any additional instructions passed for the assistant
    additional_instructions = f"<additional_instructions>{request.additional_instructions}</additional_instructions>" if request.additional_instructions else None

    async def event_stream():
        """
        Asynchronously stream responses back to the caller in JSON lines with granular events.
        """
        try:
            # Stage 1: Initialize connection
            init_data = json.dumps({
                "event": "connection_established",
                "data": "Connection established, preparing to process query...",
                "threadId": current_thread_id,
                "stage": "init"            })
            yield f"data: {init_data}\n\n"

            # Stage 2: Processing query (add small delay to make it realistic)
            await asyncio.sleep(0.1)  # Small delay to simulate processing
            processing_data = json.dumps({
                "event": "query_processing",
                "data": "Processing query and preparing LLM request...",
                "threadId": current_thread_id,
                "stage": "processing"            })
            yield f"data: {processing_data}\n\n"

            # Stage 3: Start streaming - this is where the real work begins
            # Stage 4: Stream LLM responses
            first_chunk = True
            llm_call_initiated = False
            async for partial_content in agent.invoke_stream(thread_id=current_thread_id, additional_instructions=additional_instructions):
                
                # Send LLM call initiated event only when we actually start streaming
                if not llm_call_initiated:
                    llm_init_data = json.dumps({
                        "event": "llm_call_initiated",
                        "data": "Initiating LLM call with invoke_stream...",
                        "threadId": current_thread_id,
                        "stage": "llm_init"
                    })
                    yield f"data: {llm_init_data}\n\n"
                    llm_call_initiated = True
                
                # Skip empty content
                if not partial_content.content.strip():
                    continue
                  # Send first chunk event
                if first_chunk:
                    first_chunk_data = json.dumps({
                        "event": "first_chunk_received",
                        "data": "First response chunk received, starting stream...",
                        "threadId": current_thread_id,
                        "stage": "streaming"
                    })
                    yield f"data: {first_chunk_data}\n\n"
                    first_chunk = False
                
                # Prepare the data for streaming
                data = json.dumps({
                    "event": "content_chunk",
                    "data": partial_content.content,
                    "threadId": current_thread_id,
                    "stage": "streaming"
                })

                # Stream to the caller
                yield f"data: {data}\n\n"

            # Stage 5: Stream completion
            completion_data = json.dumps({
                "event": "stream_completed",
                "data": "Stream processing completed successfully.",
                "threadId": current_thread_id,
                "stage": "completed"
            })
            yield f"data: {completion_data}\n\n"

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

    # Return text/event-stream response
    return StreamingResponse(event_stream(), media_type="text/event-stream", status_code=200)

@app.post("/shadow-sk-no-stream")
async def shadow_sk_no_stream(request: ShadowRequest):
    """
    Endpoint that receives a query, passes it to the agent, and returns a single JSON response.
    """
    agent = await get_agent()

    # Assume `request` is already an instance of ShadowRequest

    # Extract fields directly
    query = request.query  # required field, always present
    threadId = request.threadId  # required field, always present
    user_company = request.user_company  # optional
    target_account = request.target_account  # optional
    demand_stage = request.demand_stage # optional

    # Build structured parameters
    params = {
        "target_account": target_account,
        "user_company": user_company,
        "demand_stage": demand_stage
    }

    # Combine query and parameters into a single string
    combined_query = f"{query} - {params}"

    # Retrieve or create a thread ID
    if threadId:
        current_thread_id = threadId
    else:
        current_thread_id = await agent.create_thread()  # create new threadId and set as current

    # Create the user message content with the request.query
    message_user = ChatMessageContent(role=AuthorRole.USER, content=combined_query)
    await agent.add_chat_message(thread_id=current_thread_id, message=message_user)
    
    # get any additional instructions passed for the assistant
    additional_instructions = f"<additional_instructions>{request.additional_instructions}</additional_instructions>" or None

    try:
        # Collect all messages from the async iterable
        full_response = []
        async for message in agent.invoke(thread_id=current_thread_id, additional_instructions=additional_instructions):
            if message.content.strip():  # Skip empty content
                full_response.append(message.content)

        if not full_response:
            return {"error": "Empty response from the agent.", "threadId": current_thread_id}

        # Combine the collected messages into a single string
        combined_response = " ".join(full_response)

        json_response = {
            "data": combined_response,
            "threadId": current_thread_id
        }
        # Return json response
        return JSONResponse(json_response, status_code=200)

    except HTTPException as exc:
        return {"error": exc.detail}
    except Exception as e:
        logging.exception("Unexpected error during response generation.")
        return {"error": str(e)}