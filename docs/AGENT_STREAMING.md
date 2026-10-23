# Agent streaming and LangGraph compatibility

AgenticMemoryRuntime accepts a configured LangChain agent or compiled LangGraph agent that implements invoke(input, config=...). The host application remains responsible for model credentials, tools, permissions, and graph construction.

## Non-streaming (default)

```python
runtime = AgenticMemoryRuntime(user_id="user_123", agent=agent)
result = runtime.run("Summarize my project")
print(result["answer"])
```

## Opt-in streaming

Enable streaming in code when creating the runtime. There is no runtime UI toggle.

```python
runtime = AgenticMemoryRuntime(
    user_id="user_123",
    agent=agent,
    streaming=True,
)

result = None
for event in runtime.stream("Build a complete portfolio website"):
    if event["type"] == "token":
        print(event["content"], end="", flush=True)
    elif event["type"] == "result":
        result = event["result"]

if result is None:
    raise RuntimeError("The stream ended without a final result.")
```

Events have a stable shape: {"type": "token", "content": "..."} while tokens arrive, followed by {"type": "result", "result": {...}} after the runtime has recorded the final answer and checkpoint. Exceptions raised by the worker are re-raised in the consuming thread.

Token events depend on the configured model and agent propagating LangChain callbacks. The Streamlit demo configures its ChatOpenAI model with streaming=True and its runtime with streaming=True. To disable streaming there, change the streaming argument in build_agent_runtime() and the model's streaming setting together.

## 504 and provider timeouts

The Streamlit demo uses MEMORI_AGENT_TIMEOUT_SECONDS (default 180) for the model request and allows two client retries for transient request errors. Set the environment variable to a provider-supported value if needed. Streaming can improve time-to-first-token but does not guarantee that a provider or gateway 504 will disappear. Inspect provider status, request limits, model availability, and server logs when 504s persist.

Retries are configured at the model client layer; the runtime does not blindly replay a completed agent turn, which could repeat tool side effects.
