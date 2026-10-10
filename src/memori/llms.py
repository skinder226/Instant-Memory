import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .schema import MemorySchema, RetivalTypes, RetrievalPlan, MemoryDecision

load_dotenv()

openai_compatible_providers = {
    "openai", 
    "nvidia", 
    "deepseek",      # DeepSeek API (v1/chat/completions)
    "groq",          # Groq Cloud API (openai/v1)
    "mistral",       # Mistral AI API
    "together",      # Together AI
    "fireworks",     # Fireworks AI
    "openrouter",    # OpenRouter (Aggregator endpoint)
    "xai",           # xAI (Grok)
    "cohere",        # Cohere Compatibility API
    "cerebras",      # Cerebras Inference API
    "anthropic",     # If routed through a gateway or proxy layer
    "ollama",        # Local deployment / self-hosted
    "vllm",          # Local deployment / self-hosted inference engine
    "lm_studio"      # Local deployment / offline testing
}

if not os.getenv("Mem_llm_provider"):
    raise ValueError("Mem_llm_provider environment variable is not set. Please set it in your .env file.")

if os.getenv("Mem_llm_provider") == "YOUR_PROVIDER" or os.getenv("Mem_llm_provider") == "":
    raise ValueError("Mem_llm_provider environment variable is not set. Please set it in your .env file.")

mem_llm_provider = os.getenv("Mem_llm_provider")

if os.getenv("Mem_llm_provider") not in openai_compatible_providers:
    raise ValueError(f"Mem_llm_provider '{mem_llm_provider}' is not supported. Please choose from: {', '.join(openai_compatible_providers)}")


if mem_llm_provider in openai_compatible_providers:
    if not os.getenv("Mem_llm_Api"):
        raise ValueError("Mem_llm_Api environment variable is not set. Please set it in your .env file.")
    if os.getenv("Mem_llm_Api") == "YOUR_API_KEY" or os.getenv("Mem_llm_Api") == "":
        raise ValueError("Mem_llm_Api environment variable is not set. Please set it in your .env file.")
    Mem_llm_Api = os.getenv("Mem_llm_Api")

    if not os.getenv("Mem_base_url"):
        raise ValueError("Mem_base_url environment variable is not set. Please set it in your .env file.")
    if os.getenv("Mem_base_url") == "YOUR_BASE_URL" or os.getenv("Mem_base_url") == "":
        raise ValueError("Mem_base_url environment variable is not set. Please set it in your .env file.")

    Mem_base_url = os.getenv("Mem_base_url")

    if not os.getenv("Mem_model"):
        raise ValueError("Mem_model environment variable is not set. Please set it in your .env file.")
    if os.getenv("Mem_model") == "YOUR_MODEL" or os.getenv("Mem_model") == "":
        raise ValueError("Mem_model environment variable is not set. Please set it in your .env file.")

    Mem_model = os.getenv("Mem_model")

    chat = ChatOpenAI(
        model=Mem_model,
        api_key=Mem_llm_Api,
        temperature=0,
        base_url=Mem_base_url,
    )




# Function-calling structured output makes the Pydantic schema part of the
# model interface instead of asking the model to emit unconstrained JSON.
query_converter = chat.with_structured_output(RetivalTypes)
memory_extractor = chat.with_structured_output(MemorySchema)
RetrievalRouter = chat.with_structured_output(RetrievalPlan)
MemoryDecisionGate = chat.with_structured_output(MemoryDecision)
