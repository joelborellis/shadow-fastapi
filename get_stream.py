import aiohttp
import asyncio
import json
import time
from datetime import datetime


async def consume_sse(url: str, payload: str):
    """
    Connects to an SSE endpoint and prints out parsed JSON lines
    with debug information for each event type and timing.
    """
    start_time = time.time()
    last_event_time = start_time
    thread_id_printed = False
    first_content_chunk = True
    
    print(f"\n[DEBUG] Starting SSE connection at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    print(f"[DEBUG] URL: {url}")
    print(f"[DEBUG] Payload: {json.dumps(payload, indent=2)}")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            print(f"[DEBUG] Connection established, status: {response.status}")
            
            async for chunk, _ in response.content.iter_chunks():
                current_time = time.time()
                elapsed_from_start = current_time - start_time
                elapsed_from_last = current_time - last_event_time
                
                # Decode chunk into text
                text_chunk = chunk.decode("utf-8")

                # The server might send multiple lines in one chunk,
                # so we split by newlines to handle them individually
                for line in text_chunk.splitlines():
                    line = line.strip()
                    if not line:
                        # Skip empty lines
                        continue
                    
                    # Handle extra "data:" prefix if present
                    if line.startswith("data: "):
                        line = line[len("data: ") :]
                    
                    try:
                        json_data = json.loads(line)
                        event_type = json_data.get("event", "unknown")
                        content = json_data.get("data", "")
                        thread_id = json_data.get("threadId", "")
                        stage = json_data.get("stage", "")
                        
                        # Print thread_id only once
                        if thread_id and not thread_id_printed:
                            print(f"[DEBUG] Thread ID: {thread_id}")
                            thread_id_printed = True
                        
                        # Handle different event types
                        if event_type == "connection_established":
                            # Print timing and event debug info
                            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            print(f"\n[DEBUG] {timestamp} | Event: {event_type} | Stage: {stage}")
                            print(f"[DEBUG] Elapsed from start: {elapsed_from_start:.3f}s | From last event: {elapsed_from_last:.3f}s")
                            print(f"[EVENT] ✓ Connection established")
                            
                        elif event_type == "query_processing":
                            # Print timing and event debug info
                            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            print(f"\n[DEBUG] {timestamp} | Event: {event_type} | Stage: {stage}")
                            print(f"[DEBUG] Elapsed from start: {elapsed_from_start:.3f}s | From last event: {elapsed_from_last:.3f}s")
                            print(f"[EVENT] ⚙️ Processing query...")
                            
                        elif event_type == "llm_call_initiated":
                            # Print timing and event debug info
                            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            print(f"\n[DEBUG] {timestamp} | Event: {event_type} | Stage: {stage}")
                            print(f"[DEBUG] Elapsed from start: {elapsed_from_start:.3f}s | From last event: {elapsed_from_last:.3f}s")
                            print(f"[EVENT] 🚀 LLM call initiated")
                            
                        elif event_type == "first_chunk_received":
                            # Print timing and event debug info
                            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            print(f"\n[DEBUG] {timestamp} | Event: {event_type} | Stage: {stage}")
                            print(f"[DEBUG] Elapsed from start: {elapsed_from_start:.3f}s | From last event: {elapsed_from_last:.3f}s")
                            print(f"[EVENT] 📥 First chunk received, starting content stream...")
                            print(f"[DEBUG] Content streaming started")
                            
                        elif event_type == "content_chunk":
                            # Print debug info only for the first content chunk
                            if first_content_chunk:
                                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                                print(f"\n[DEBUG] {timestamp} | Event: {event_type} | Stage: {stage}")
                                print(f"[DEBUG] Elapsed from start: {elapsed_from_start:.3f}s | From last event: {elapsed_from_last:.3f}s")
                                first_content_chunk = False
                            
                            # Print content directly without debug prefixes
                            if content:
                                print(content, end="", flush=True)
                                    
                        elif event_type == "stream_completed":
                            total_time = time.time() - start_time
                            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            print(f"\n[DEBUG] {timestamp} | Event: {event_type} | Stage: {stage}")
                            print(f"[DEBUG] Elapsed from start: {elapsed_from_start:.3f}s | From last event: {elapsed_from_last:.3f}s")
                            print(f"[DEBUG] Content streaming completed")
                            print(f"[EVENT] ✅ Stream completed successfully")
                            print(f"[DEBUG] Total processing time: {total_time:.3f}s")
                            
                        elif event_type == "error":
                            # Print timing and event debug info
                            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            print(f"\n[DEBUG] {timestamp} | Event: {event_type} | Stage: {stage}")
                            print(f"[DEBUG] Elapsed from start: {elapsed_from_start:.3f}s | From last event: {elapsed_from_last:.3f}s")
                            print(f"[EVENT] ❌ Error occurred: {content}")
                            
                        else:
                            # Print timing and event debug info
                            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            print(f"\n[DEBUG] {timestamp} | Event: {event_type} | Stage: {stage}")
                            print(f"[DEBUG] Elapsed from start: {elapsed_from_start:.3f}s | From last event: {elapsed_from_last:.3f}s")
                            print(f"[EVENT] Unknown event type: {event_type}")
                            if content:
                                print(f"[CONTENT] {content}")
                        
                        last_event_time = current_time
                        
                    except json.JSONDecodeError:
                        print(f"[ERROR] Could not parse JSON: {line}")

    total_time = time.time() - start_time
    print(f"\n[DEBUG] SSE connection closed. Total session time: {total_time:.3f}s")
    return json_data.get("threadId", "") if 'json_data' in locals() else ""


async def main():
    threadId = ""

    print("=" * 60)
    print("Shadow Streaming Client with Event Debugging")
    print("=" * 60)

    while True:
        # Get user query
        query = input(f"\nAsk Shadow: ")
        if query.lower() == "exit":
            print("\n[DEBUG] Exiting client...")
            exit(0)

        # Point this to your actual SSE endpoint
        # url = "https://shadow-endpoint-k33pqykzy3hqo-function-app.azurewebsites.net/shadow-sk"
        url = "http://localhost:7071/shadow-sk"  # Use streaming endpoint

        # Construct request payload
        payload = {
            "query": query,
            "threadId": threadId,
            "additional_instructions": "Output your response in markdown format",
            "user_company": "MultiPlan",
            "target_account": "Labcorp",
            "demand_stage": "Interest",
        }
        
        # Call consume_sse which will handle the streaming events
        print(f"\n[DEBUG] Initiating request...")
        threadId = await consume_sse(url, payload)
        print(f"\n[DEBUG] Request completed. Thread ID for next request: {threadId}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())