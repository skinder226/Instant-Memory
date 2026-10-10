import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# Fixed import here:
from langchain_core.messages import SystemMessage, HumanMessage
from memori import memori

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
print("🤖 LangChain Chatbot initialized! Type 'exit' or 'quit' to stop.\n")

# 2. Continuous interaction loop
while True:
    try:
        user_input = input("You: ")
        
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        memory_response = memori("user_125", user_input)
        print("AI: ", end="", flush=True)
        
        # 3. Construct LangChain Message structures safely
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=f"Here are the retrieved memories: {memory_response}"),
            HumanMessage(content=user_input)
        ]
        
        # 4. Stream chunks directly
        full_response = ""
        for chunk in llm.stream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
        print("\n") 

    except KeyboardInterrupt:
        print("\nSession interrupted. Goodbye!")
        break
    except Exception as e:
        print(f"\nAn error occurred: {e}\n")
