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

from typing import Optional, AsyncGenerator

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
        shadow_plugin = ShadowInsightsPlugin(
            search_shadow_client, search_customer_client, search_user_client
        )
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
            id=ASSISTANT_ID, kernel=kernel, ai_model_id="gpt-4.1-mini"
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
    """
    agent = await get_agent()

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
        current_thread_id = threadId
    else:
        current_thread_id = (
            await agent.create_thread()
        )  # create new threadId and set as current

    # Create the user message content with the request.query
    message_user = ChatMessageContent(role=AuthorRole.USER, content=combined_query)
    # Add the user message to the agent
    await agent.add_chat_message(thread_id=current_thread_id, message=message_user)
    # get any additional instructions passed for the assistant
    additional_instructions = (
        f"<additional_instructions>{request.additional_instructions}</additional_instructions>"
        if request.additional_instructions
        else None
    )

    try:
        # Stage 1: Invoke stream on the agent
        #first_chunk = True

        # Stage 2: Send connection established message
        #init_data = json.dumps(
        #    {
        #        "event": "connection_established",
        #        "data": "Connection established, preparing to process query...",
        #        "threadId": current_thread_id,
        #        "stage": "init",
        #    }
        #)
        #yield f"data: {init_data}\n\n"

        # Force a small delay to ensure the init event is sent before processing
        #await asyncio.sleep(0.05)

        # Stage 2: Processing query - send immediately
        #processing_data = json.dumps(
        #    {
        #        "event": "query_processing",
        #        "data": "Processing query and preparing LLM request...",
        #        "threadId": current_thread_id,
        #        "stage": "processing",
        #    }
        #)
        #yield f"data: {processing_data}\n\n"

        # Force a small delay to ensure the processing event is sent before LLM call
        #await asyncio.sleep(0.05)

        # Stage 3: Invoke stream on the agent (after initial yields)
        result = agent.invoke_stream(
            thread_id=current_thread_id, additional_instructions=additional_instructions
        )
        
        async for partial_content in result:
            # Skip empty content
            if not partial_content.content.strip():
                continue

            # Send first chunk event
            #if first_chunk:
            #    first_chunk_data = json.dumps(
            #        {
            #            "event": "first_chunk_received",
            #            "data": "First response chunk received, starting stream...",
            #            "threadId": current_thread_id,
            #            "stage": "streaming",
            #        }
            #    )
            #    yield f"data: {first_chunk_data}\n\n"
            #    first_chunk = False

            # Prepare the data for streaming
            data = json.dumps(
                {
                    "event": "content_chunk",
                    "data": partial_content.content,
                    "threadId": current_thread_id,
                    "stage": "streaming",
                }
            )

            # Stream to the caller
            yield f"data: {data}\n\n"

        # Stage 5: Stream completion
        #completion_data = json.dumps(
        #    {
        #        "event": "stream_completed",
        #        "data": "Stream processing completed successfully.",
        #        "threadId": current_thread_id,
        #        "stage": "completed",
        #    }
        #)
        #yield f"data: {completion_data}\n\n"

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