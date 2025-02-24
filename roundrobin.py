import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core import CancellationToken
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
    system_message="You are a helpful AI assistant which checks grammer, readability, clarity and ensures neutrality and fairness who adds a postive impactful ending to the news article. Feel free to expand the article as needed. Respond with 'APPROVE' only when you are fully done",
)

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("APPROVE")

# Create a team with the primary and critic agents.
team = RoundRobinGroupChat([News_Reporter, News_Editor], termination_condition=text_termination)

# Define the main asynchronous function
async def main():
    await Console(
        #team.run_stream(task="The rise of AI-driven misinformation in politics")
        team.run_stream(task="Is remote work the future, or are we losing workplace culture?")
    )  # Stream the messages to the console.

# Run the asynchronous function
if __name__ == "__main__":
    asyncio.run(main())