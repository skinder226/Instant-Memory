import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .schema import MemorySchema, RetivalTypes, RetrievalPlan, MemoryDecision

load_dotenv()

chat = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0,
    base_url="https://integrate.api.nvidia.com/v1",
)

# Function-calling structured output makes the Pydantic schema part of the
# model interface instead of asking the model to emit unconstrained JSON.
query_converter = chat.with_structured_output(RetivalTypes)
memory_extractor = chat.with_structured_output(MemorySchema)
RetrievalRouter = chat.with_structured_output(RetrievalPlan)
MemoryDecisionGate = chat.with_structured_output(MemoryDecision)
