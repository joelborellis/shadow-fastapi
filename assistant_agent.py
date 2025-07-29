import asyncio
from typing import Annotated

from semantic_kernel.agents import AssistantAgentThread, AzureAssistantAgent, OpenAIAssistantAgent
from semantic_kernel.connectors.ai.open_ai import AzureOpenAISettings, OpenAISettings
from semantic_kernel.contents import AuthorRole, ChatMessageContent, FunctionCallContent, FunctionResultContent
from semantic_kernel.functions import kernel_function


# Define a sample plugin for the sample
class MenuPlugin:
    """A sample Menu Plugin used for the concept sample."""

    @kernel_function(description="Provides a list of specials from the menu.")
    def get_specials(self) -> Annotated[str, "Returns the specials from the menu."]:
        return """
        Special Soup: Clam Chowder
        Special Salad: Cobb Salad
        Special Drink: Chai Tea
        """

    @kernel_function(description="Provides the price of the requested menu item.")
    def get_item_price(
        self, menu_item: Annotated[str, "The name of the menu item."]
    ) -> Annotated[str, "Returns the price of the menu item."]:
        return "$9.99"


# This callback function will be called for each intermediate message,
# which will allow one to handle FunctionCallContent and FunctionResultContent.
# If the callback is not provided, the agent will return the final response
# with no intermediate tool call steps.
async def handle_streaming_intermediate_steps(message: ChatMessageContent) -> None:
    for item in message.items or []:
        if isinstance(item, FunctionResultContent):
            print(f"Function Result:> {item.result} for function: {item.name}")
        elif isinstance(item, FunctionCallContent):
            print(f"Function Call:> {item.name} with arguments: {item.arguments}")
        else:
            print(f"{item}")


async def main():
    # Create the client using Azure OpenAI resources and configuration
    client = OpenAIAssistantAgent.create_client(ai_model_id="gpt-4o")

    # Define the assistant definition
    definition = await client.beta.assistants.retrieve("asst_OE81y9Pa0e6w95ViwH6WTYrv")

    # Create the OpenAIAssistantAgent instance using the client and the assistant definition and the defined plugin
    agent = OpenAIAssistantAgent(
        client=client,
        definition=definition,
        plugins=[MenuPlugin()],
    )

    # Create a new thread for use with the assistant
    # If no thread is provided, a new thread will be
    # created and returned with the initial response
    thread: AssistantAgentThread = None

    user_inputs = [
        "Hello",
        "What is the special soup?",
        "What is the special drink?",
        "How much is that?",
        "Thank you",
    ]

    try:
        for user_input in user_inputs:
            print(f"# {AuthorRole.USER}: '{user_input}'")

            first_chunk = True
            async for response in agent.invoke_stream(
                messages=user_input,
                thread=thread,
                on_intermediate_message=handle_streaming_intermediate_steps,
            ):
                thread = response.thread
                if first_chunk:
                    print(f"# {response.name}: ", end="", flush=True)
                    first_chunk = False
                print(response.content, end="", flush=True)
            print()
    finally:
        await thread.delete() if thread else None


if __name__ == "__main__":
    asyncio.run(main())