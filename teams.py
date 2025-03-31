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

# Create the primary agent.
Story_writer = AssistantAgent(
    "Story_writer",
    model_client=az_model_client,
    system_message="You are a helpful AI assistant which write the story for kids. Keep the story short",
)

# Create the critic agent.
Story_reviewer = AssistantAgent(
    "Story_reviewer",
    model_client=az_model_client,
    system_message="You are a helpful AI assistant which Provides constructive feedback on Kids stories to add a postive impactful ending. Respond with 'APPROVE' to when your feedbacks are addressed.",
)

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("APPROVE")

# Create a team with the primary and critic agents.
team = RoundRobinGroupChat([Story_writer, Story_reviewer], termination_condition=text_termination)

# Define the main asynchronous function
async def main():
    await Console(
        team.run_stream(task="write a story on lion")
    )  # Stream the messages to the console.

# Run the asynchronous function
if __name__ == "__main__":
    asyncio.run(main())