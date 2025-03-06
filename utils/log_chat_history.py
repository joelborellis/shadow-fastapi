import logging
from functools import wraps
import json

# Configure module-level logger
logger = logging.getLogger("log_chat_history")
logger.setLevel(logging.INFO)

# Define the decorator function
def log_chat_model_dump(chat, parser_function=None):
    """
    Decorator to log the value of `chat.model_dump_json()` each time the decorated function runs.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)  # Don't await here yet to keep it a coroutine
            try:
                async for value in result:
                    #print(value)
                    yield value  # Yield the streamed data which is the response from the result of the function
            finally:
                # Log the value of `chat.model_dump_json()`
                if hasattr(chat, 'model_dump_json'):
                    json_data = (json.loads(chat.model_dump_json()))
                    logging.info(json.dumps(json_data))
                # Pass to parser_function if provided
                if parser_function and callable(parser_function):
                        try:
                            jsonl_row = parser_function(json_data)
                            logger.info(f"{jsonl_row}")
                        except Exception as e:
                            logger.error(f"Error in parsing function: {e}")
                            print(f"Error in parsing function: {e}")
        return wrapper

    return decorator

import json

def extract_assistant_shadow_text(data):
    """
    Extracts text and relevant metadata from a JSON structure containing messages, then
    merges them into a single JSONL row (string).

    Specifically:
    1. Gathers all `text` content from messages where role="user".
    2. Gathers all `text` content from messages where role="assistant" and name="Shadow".
    3. Gathers specific metadata (arguments, function_name, plugin_name) from messages where role="tool" and name="Shadow".

    Args:
        data (dict): The JSON structure containing messages.

    Returns:
        str: A single JSONL row containing all extracted data.
    """

    user_texts = []
    assistant_shadow_texts = []
    tool_shadow_data = []
    token_data = []

    messages = data.get("messages", [])
    for message in messages:
        role = message.get("role")
        name = message.get("name", "")
        finish_reason = message.get("finish_reason", "")
        items = message.get("items", [])

        if role == "user":
            for item in items:
                text = item.get("text")
                if text:
                    user_texts.append(text)

        elif role == "assistant" and name == "Shadow":
            for item in items:
                text = item.get("text")
                if text:
                    assistant_shadow_texts.append(text)
                # 4. If finish_reason == "tool_calls", capture usage info
                if finish_reason == "tool_calls":
                    usage_info = message.get("metadata", {}).get("usage", {})
                    token_data.append(usage_info)

        elif role == "tool" and name == "Shadow":
            for item in items:
                metadata = item.get("metadata", {})
                arguments = metadata.get("arguments")
                function_name = item.get("function_name")
                plugin_name = item.get("plugin_name")
                if arguments and function_name and plugin_name:
                    tool_shadow_data.append({
                        "plugin_name": plugin_name,
                        "function_name": function_name,
                        "arguments": arguments
                    })

    # Combine the three lists into a single JSON object
    combined_data = {
        "user": user_texts,
        "assistant": assistant_shadow_texts,
        "tool_call": tool_shadow_data,
        "usage": token_data,
    }

    # Convert to a JSONL (one JSON object per line) string
    jsonl_row = json.dumps(combined_data)

    return jsonl_row