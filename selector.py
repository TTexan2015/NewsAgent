from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

# # Create the token provider
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

API_KEY = os.getenv("api_key")
Model_Name = os.getenv("model-name")
Api_Version = os.getenv("api-version")
Azure_Endpoint = os.getenv("azure_endpoint")

az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=Model_Name,
    model=Model_Name,
    api_version=Api_Version,
    azure_endpoint=Azure_Endpoint,
    api_key=API_KEY
)

planning_agent = AssistantAgent(
    "PlanningAgent",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=az_model_client,
    system_message="""
    You are a planning agent.
    Your job is to break down complex tasks into smaller, manageable subtasks.
    Your team members are:
        News_Reporter: Writes news article.
        News_Editor: Checks and Provides constructive feedback. It doesn't write the article, only provide feedback and improvements.
        Headline_Generator: Finally, adds the moral to the story.

    You only plan and delegate tasks - you do not execute them yourself. You can engage team members multiple times so that a perfect story is provided.

    When assigning tasks, use this format:
    1. <agent> : <task>

    After all tasks are complete, summarize the findings and end with "TERMINATE".
    """,
)

# Create the News Reporter agent.
News_Reporter = AssistantAgent(
    "News_Reporter_agent",
    model_client=az_model_client,
    system_message="You are a helpful AI assistant which write news article based on given facts. Keep the article short",
)

# Create the Editor agent.
News_Editor = AssistantAgent(
    "News_Editor_agent",
    model_client=az_model_client,
    system_message="You are a helpful AI assistant which checks grammer, readability, clarity and ensures neutrality and fairness and provides feedback. You do not write the article",
)

# Story Moral Agent.
Headline_Generator = AssistantAgent(
    "Headline_Generator_agent",
    model_client=az_model_client,
    system_message="You create engaging headlines",
)

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=10)
termination = text_mention_termination | max_messages_termination

team = SelectorGroupChat(
    [planning_agent, News_Reporter, News_Editor, Headline_Generator],
    model_client=az_model_client,
    termination_condition=termination,
)

# Define the main asynchronous function
async def main():
    await Console(
        team.run_stream(task="Is remote work the future, or are we losing workplace culture?")
    )  # Stream the messages to the console.

# Run the asynchronous function
if __name__ == "__main__":
    asyncio.run(main())