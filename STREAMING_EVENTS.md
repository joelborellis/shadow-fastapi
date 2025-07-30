# Shadow FastAPI Streaming Events

## Overview

The Shadow FastAPI application now supports real-time streaming of intermediate events during agent processing. This allows clients to receive live updates about function calls, function results, and other processing steps as they happen.

## Event Types

The streaming endpoint now emits the following event types:

### 1. `thread_info`
Emitted when the agent starts processing, contains thread and agent information.
```json
{
  "type": "thread_info",
  "agent_name": "ShadowAgent",
  "thread_id": "thread_abc123"
}
```

### 2. `function_call`
Emitted when the agent calls a tool/function.
```json
{
  "type": "function_call",
  "function_name": "search_shadow",
  "arguments": {"query": "sales performance", "filters": {...}}
}
```

### 3. `function_result`
Emitted when a tool/function returns its result.
```json
{
  "type": "function_result",
  "function_name": "search_shadow",
  "result": "Found 15 relevant insights about sales performance..."
}
```

### 4. `content`
Emitted for actual response content from the agent.
```json
{
  "type": "content",
  "content": "Based on the search results, here are the key insights..."
}
```

### 5. `intermediate`
Emitted for other intermediate processing events.
```json
{
  "type": "intermediate",
  "content": "Processing request..."
}
```

### 6. `stream_complete`
Emitted when the stream is finished.
```json
{
  "type": "stream_complete"
}
```

### 7. `error`
Emitted when an error occurs.
```json
{
  "type": "error",
  "error": "Error message here"
}
```

## Usage

### Client Implementation

```javascript
// Example client-side implementation
async function streamShadowEvents(query, threadId) {
    const response = await fetch('/shadow-sk', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: query,
            threadId: threadId,
            user_company: "Acme Corp",
            target_account: "BigClient Inc"
        })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
            if (line.startsWith('event:')) {
                const eventType = line.substring(6).trim();
                console.log('Event Type:', eventType);
            } else if (line.startsWith('data:')) {
                const data = JSON.parse(line.substring(5).trim());
                console.log('Event Data:', data);
                
                // Handle different event types
                switch (data.type) {
                    case 'function_call':
                        console.log(`🔧 Calling function: ${data.function_name}`);
                        break;
                    case 'function_result':
                        console.log(`✅ Function result: ${data.function_name}`);
                        break;
                    case 'content':
                        console.log(`💬 Agent response: ${data.content}`);
                        break;
                    case 'stream_complete':
                        console.log('🏁 Stream finished');
                        return;
                }
            }
        }
    }
}
```

### Python Client Example

See `test_streaming_events.py` for a complete Python client example using aiohttp.

## Technical Implementation

The streaming is implemented using:
- **AsyncGenerator**: For streaming responses back to the client
- **asyncio.Queue**: For passing events between the intermediate callback and main stream
- **Server-Sent Events (SSE)**: For real-time event delivery to the client
- **Background Task**: The agent processing runs in a separate task while events are streamed

## Benefits

1. **Real-time Feedback**: Clients can show live progress of agent processing
2. **Transparency**: Users can see which tools are being called and their results
3. **Better UX**: Immediate feedback instead of waiting for final response
4. **Debugging**: Easier to track what the agent is doing step by step
