import logging
import os
import time
from rich.logging import RichHandler
from rich.console import Console
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# Fixed import here:
from langchain_core.messages import SystemMessage, HumanMessage
from memori import memori

# Readable, colored logs for the interactive test client. Keep low-level Mongo
# command timings and raw HTTP request lines out of the normal console output.
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False, markup=True)],
    force=True,
)
logging.getLogger("memori.mongodb_connection").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Load variables from your local .env file securely
load_dotenv()

# 1. Initialize the ChatOpenAI client configured for NVIDIA NIM
# llm = ChatOpenAI(
#     base_url="https://integrate.api.nvidia.com/v1",
#     api_key=os.getenv("NVIDIA_API_KEY"), 
#     model="nvidia/nemotron-3-ultra-550b-a55b",
#     temperature=0.5,
#     streaming=True
# )

llm =   ChatGroq(
    api_key=os.getenv("Mem_llm_Api"),
    model=os.getenv("Mem_model"),
    temperature=0.5,
    streaming=True
)
console.print("[bold cyan]Instant-Memory Chatbot[/bold cyan]  [dim]Type 'exit' or 'quit' to stop.[/dim]\n")

# 2. Continuous interaction loop
while True:
    try:
        user_input = input("You: ")
        
        if user_input.strip().lower() in ['exit', 'quit']:
            console.print("[yellow]Goodbye![/yellow]")
            break
            
        if not user_input.strip():
            continue
            
        retrieval_started = time.perf_counter()
        memory_response = memori("user_125", user_input, source="user")
        # The package API returns a structured result; pass only retrieved memories to the LLM.
        if isinstance(memory_response, dict):
            memory_response = memory_response.get("retrieval", [])
        retrieval_seconds = time.perf_counter() - retrieval_started
        console.print(f"[bold cyan]🧠 MEMORY RETRIEVAL[/bold cyan] [green]{retrieval_seconds:.3f}s[/green]  [dim]memories={len(memory_response) if isinstance(memory_response, list) else 0}[/dim]")
        console.print("[bold magenta]AI[/bold magenta] ", end="")
        
        # 3. Construct LangChain Message structures safely
        messages = [
            SystemMessage(content=(
                "You are a helpful assistant. For questions about the user's personal life, "
                "use retrieved memories as the source of truth. If relevant memories are provided, "
                "answer directly from them and do not claim you have no memory. If the retrieved "
                "memories do not contain the answer, say the available memories do not establish it; "
                "do not invent relationships or facts."
            )),
            HumanMessage(content=f"Here are the retrieved memories: {memory_response}"),
            HumanMessage(content=user_input)
        ]
        
        # 4. Stream chunks directly
        response_started = time.perf_counter()
        full_response = ""
        for chunk in llm.stream(messages):
            if chunk.content:
                console.print(chunk.content, end="")
                full_response += chunk.content
        response_seconds = time.perf_counter() - response_started
        console.print(f"\n[bold magenta]⚡ AI RESPONSE[/bold magenta] [green]{response_seconds:.3f}s[/green]")
        console.print(f"[bold]⏱ TOTAL TURN[/bold] [green]{retrieval_seconds + response_seconds:.3f}s[/green]\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted. Goodbye![/yellow]")
        break
    except Exception as e:
        console.print(f"\n[bold red]ERROR[/bold red] {e}\n")
