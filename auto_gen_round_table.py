# auto_gen_round_table_parallel.py
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio

# Ollama via OpenAI-compatible API
model_client = OpenAIChatCompletionClient(
    model="llama3.2:latest",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": "llama",
        "structured_output": False,
    },
    max_tokens=200
)

# Define agents
agents = [
    AssistantAgent(
        name="Albert_Einstein",
        model_client=model_client,
        system_message="You are Albert Einstein, theoretical physicist. Curious, playful, love analogies. Keep responses concise."
    ),
    AssistantAgent(
        name="Leonardo_da_Vinci",
        model_client=model_client,
        system_message="You are Leonardo da Vinci, Renaissance polymath. Observant, imaginative. Keep responses concise."
    ),
    AssistantAgent(
        name="Marie_Curie",
        model_client=model_client,
        system_message="You are Marie Curie, precise scientist. Emphasize facts. Keep responses concise."
    ),
    AssistantAgent(
        name="Socrates",
        model_client=model_client,
        system_message="You are Socrates, classical philosopher. Ask probing questions. Keep responses concise."
    ),
    AssistantAgent(
        name="Adolf_Hitler",
        model_client=model_client,
        system_message="You are Adolf Hitler, the historical figure and resentful dictator of Nazi Germany. Speak in the context of historical events, ideology, and decisions. Keep responses concise."
    ),
]

# Round robin team
termination = MaxMessageTermination(max_messages=10)
team = RoundRobinGroupChat(agents, termination_condition=termination)

async def main():
    topic = "The future of humanity shaped by technology and science"
    print("=== Round Table Discussion ===\n")
    print(f"Topic: {topic}\n")
    print("=" * 50 + "\n")

    # Run the discussion
    result = await team.run(task=topic)

    # Extract and display messages from the result
    if hasattr(result, 'messages') and result.messages:
        for msg in result.messages:
            if hasattr(msg, 'source') and hasattr(msg, 'content'):
                print(f"{msg.source}: {msg.content}\n")

    print("=" * 50)
    print("=== Discussion Complete ===")

if __name__ == "__main__":
    asyncio.run(main())