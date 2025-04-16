import aiohttp
import asyncio
import json


async def consume_sse(url: str, payload: str):
    """
    Connects to an SSE endpoint and prints out parsed JSON lines
    character by character (to simulate typing).
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            async for chunk, _ in response.content.iter_chunks():
                # Decode chunk into text
                text_chunk = chunk.decode("utf-8")
                # Initialize a flag to track if thread_id has been printed
                thread_id_printed = False

                # The server might send multiple lines in one chunk,
                # so we split by newlines to handle them individually
                for line in text_chunk.splitlines():
                    line = line.strip()
                    if not line:
                        # Skip empty lines
                        continue
                    # If we reach here, line should be JSON (e.g. data: {"data": "This is some content", "thread_id": 1234}).
                    try:
                        # Handle extra "data:" prefix if present
                        if line.startswith("data: "):
                            line = line[len("data: ") :]

                        json_data = json.loads(line)
                        content = json_data.get("data", "")
                        thread_id = json_data.get("threadId", "")

                        # Print thread_id only once
                        if thread_id and not thread_id_printed:
                            print(f"Thread id: {thread_id}")
                            thread_id_printed = True

                        if content:
                            # Print line by line
                            for line in content:
                                print(line, end="", flush=True)
                                # Adjust sleep time to control the "typing" speed
                                #await asyncio.sleep(0.01)
                    except json.JSONDecodeError:
                        print("Could not parse JSON:", line)

    return thread_id


async def main():

    threadId = ""

    while True:
        # Get user query
        query = input(f"\nAsk Shadow: ")
        if query.lower() == "exit":
            exit(0)

        # Point this to your actual SSE endpoint
        #url = "https://shadow-endpoint-k33pqykzy3hqo-function-app.azurewebsites.net/shadow-sk-no-stream"
        url = "http://localhost:7071/shadow-sk-no-stream"

        # Construct request payload
        payload = {
            "query": query,
            "threadId": threadId,
            "additional_instructions": "Output your response in html format",
            "user_company": "MultiPlan",
            "target_account": "NeoGenomics",
            "demand_stage": "Interest",
        }  # thread_id will be empty first time
        # call consume what will create the streaming like output
        threadId = await consume_sse(url, payload)


if __name__ == "__main__":
    asyncio.run(main())