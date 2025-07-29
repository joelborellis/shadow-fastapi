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


# This callback function will be called for each intermediate message,
# which will allow one to handle FunctionCallContent and FunctionResultContent.
# If the callback is not provided, the agent will return the final response
# with no intermediate tool call steps.
async def handle_streaming_intermediate_steps(message: ChatMessageContent) -> None:
    for item in message.items or []:
        # Force a small delay to ensure the init event is sent before processing
        await asyncio.sleep(0.05)
        if isinstance(item, FunctionResultContent):
            print(f"Function Result:> {item.result} for function: {item.name}")
        elif isinstance(item, FunctionCallContent):
            print(f"Function Call:> {item.name} with arguments: {item.arguments}")
        else:
            print(f"{item}")


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
            "asst_3PzZuWqDfgCXAqqAtiZmLvdU"
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
        current_thread_id: AssistantAgentThread = None

    # Create the user message content with the request.query
    # message_user = ChatMessageContent(role=AuthorRole.USER, content=combined_query)
    # Add the user message to the agent
    # await agent.add_chat_message(thread_id=current_thread_id, message=message_user)
    # get any additional instructions passed for the assistant
    additional_instructions = (
        f"<additional_instructions>{request.additional_instructions}</additional_instructions>"
        if request.additional_instructions
        else None
    )

    try:
        first_chunk = True
        async for response in agent.invoke_stream(
            messages=combined_query,
            thread_id=current_thread_id,
            additional_instructions=additional_instructions,
            on_intermediate_message=handle_streaming_intermediate_steps,
        ):

            if first_chunk:
                print(f"# {response.name}: ", end="", flush=True)
                first_chunk = False
            print(response.content, end="", flush=True)
        print()

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
