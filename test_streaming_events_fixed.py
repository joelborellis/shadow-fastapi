"""
Test script to demonstrate the streaming events from the modified FastAPI endpoint.
This shows how a client would receive the different types of events.
"""

import asyncio
import aiohttp
import json

async def test_streaming_events():
    """Test the streaming endpoint and display different event types."""
    
    url = "http://localhost:7071/shadow-sk"  # Adjust URL as needed
    
    # Test payload
    payload = {
        "query": "Hello Shadow, what does my company do?",
        "threadId": "",
        "additional_instructions": "Focus on key metrics",
        "user_company": "North Highland",
        "target_account": "Allina health",
        "demand_stage": "Interest"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    print("[START] Starting to receive events...\n")
                    
                    # Track content streaming state
                    content_buffer = ""
                    content_started = False
                    
                    async for line in response.content:
                        line_str = line.decode('utf-8').strip()
                        
                        if line_str.startswith('event:'):
                            event_type = line_str.split(':', 1)[1].strip()
                            print(f"\n[EVENT] {event_type}")
                            
                        elif line_str.startswith('data:'):
                            data_str = line_str.split(':', 1)[1].strip()
                            try:
                                data = json.loads(data_str)
                                
                                # Handle different event types
                                if data.get('type') == 'function_call':
                                    # Finish any pending content display
                                    if content_buffer and content_started:
                                        print()  # New line to finish content
                                        content_buffer = ""
                                        content_started = False
                                    
                                    print(f"[FUNC CALL] {data['function_name']}")
                                    print(f"   Arguments: {data['arguments']}")
                                    
                                elif data.get('type') == 'function_result':
                                    print(f"[FUNC RESULT] {data['function_name']}")
                                    print(f"   Result: {data['result'][:100]}...")  # Truncate long results
                                    
                                elif data.get('type') == 'content':
                                    # Stream content horizontally
                                    if not content_started:
                                        print("[CONTENT] ", end="", flush=True)
                                        content_started = True
                                    
                                    content_chunk = data['content']
                                    content_buffer += content_chunk
                                    print(content_chunk, end="", flush=True)
                                    
                                elif data.get('type') == 'thread_info':
                                    print(f"[THREAD] Agent={data['agent_name']}, Thread={data['thread_id']}")
                                    
                                elif data.get('type') == 'intermediate':
                                    # Finish any pending content display
                                    if content_buffer and content_started:
                                        print()  # New line to finish content
                                        content_buffer = ""
                                        content_started = False
                                    
                                    print(f"[INTERMEDIATE] {data['content']}")
                                    
                                elif data.get('type') == 'intermediate_error':
                                    print(f"[INTERMEDIATE ERROR] {data['error']} (Item type: {data['item_type']})")
                                    
                                elif data.get('type') == 'stream_complete':
                                    # Finish any pending content display
                                    if content_buffer and content_started:
                                        print()  # New line to finish content
                                    print("\n[STREAM COMPLETE]")
                                    break
                                    
                            except json.JSONDecodeError:
                                print(f"[ERROR] Invalid JSON: {data_str}")
                                
                        # Only print separator for non-content events
                        if not line_str.startswith('data:') or ('content' not in data_str and 'stream_complete' not in data_str):
                            print("-" * 30)
                        
                else:
                    print(f"[ERROR] {response.status} - {await response.text()}")
                    
    except Exception as e:
        print(f"[CONNECTION ERROR] {e}")
        print("Make sure the FastAPI server is running on the expected port.")

if __name__ == "__main__":
    print("Testing Shadow FastAPI Streaming Events")
    print("=" * 50)
    asyncio.run(test_streaming_events())
