import fastapi
from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import json
import os
import logging

from tools.searchshadow import SearchShadow
from tools.searchcustomer import SearchCustomer

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


# Instantiate search clients as singletons (if they are thread-safe or handle concurrency internally)
search_shadow_client = SearchShadow()
search_customer_client = SearchCustomer()

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
        shadow_plugin = ShadowInsightsPlugin(search_shadow_client, search_customer_client)
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

    if request.thread_id:
        # thread_id is not empty; retrieve it
        current_thread_id = request.thread_id
    else:
        # thread_id is empty; create a new one
        current_thread_id = await agent.create_thread()

    # Create the user message content with the request.query
    message_user = ChatMessageContent(role=AuthorRole.USER, content=request.query)
    # Add the user message to the agent
    await agent.add_chat_message(thread_id=current_thread_id, message=message_user)
    
    # get any additional instructions passed for the assistant
    additional_instructions = request.additional_instructions or None

    async def event_stream():
        """
        Asynchronously stream responses back to the caller in JSON lines.
        """
        try:
            # Open a file in write mode
            # with open("stream_output.txt", "w") as file:
                async for partial_content in agent.invoke_stream(thread_id=current_thread_id, additional_instructions=additional_instructions):
                    
                    # Skip empty content
                    if not partial_content.content.strip():
                        continue
                    
                    # Prepare the data for streaming and saving
                    data = json.dumps({
                        "data": partial_content.content,
                        "threadId": current_thread_id
                    })
                    # Write to file
                    # file.write(f"{data}\n")

                    # Stream to the caller
                    yield f"data: {data}\n\n"

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

    # Retrieve or create a thread ID
    if request.threadId:
        current_thread_id = request.threadId
    else:
        current_thread_id = await agent.create_thread()

    # Create the user message content with the request.query
    message_user = ChatMessageContent(role=AuthorRole.USER, content=request.query)
    await agent.add_chat_message(thread_id=current_thread_id, message=message_user)
    
    # get any additional instructions passed for the assistant
    additional_instructions = request.additional_instructions or None

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