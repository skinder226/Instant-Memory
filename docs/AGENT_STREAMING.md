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

The Streamlit demo uses these environment settings:

- `MEMORI_AGENT_TIMEOUT_SECONDS`: per-model-request timeout in seconds (default `180`).
- `MEMORI_AGENT_MAX_RETRIES`: model-client retries for transient request errors (default `2`).

For example, in `.env`:

```dotenv
MEMORI_AGENT_TIMEOUT_SECONDS=240
MEMORI_AGENT_MAX_RETRIES=3
```

Restart Streamlit after changing environment variables. These are per-request settings, not a hard deadline for the whole agent turn. A turn can make multiple model requests and tool calls, so total duration can exceed the configured timeout.

Studio now writes a dedicated `[memori.studio]` log line and full exception traceback to the Streamlit terminal when a turn fails. The log includes user/session/task identifiers, elapsed time, exception type, and how many response characters arrived before failure; it does not log API keys. The UI distinguishes likely timeout/gateway failures from other exceptions and preserves any partial streamed text, marked as interrupted rather than successful.

Streaming can improve time-to-first-token but cannot guarantee that a provider or gateway 502/503/504 will disappear. Check the provider's status page, model availability, account limits, and request IDs if exposed by the provider. A larger timeout only helps when the provider eventually responds; it does not repair an outage or invalid model configuration.

Retries are configured at the model-client layer. The runtime does not blindly replay a whole agent turn after an exception, because it may already have invoked tools or performed side effects. If the provider is down, Studio will report the failure clearly rather than claim it can always produce an answer.
