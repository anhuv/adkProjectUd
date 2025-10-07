from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm



root_agent = Agent(
    model=LiteLlm(model="openai/nvidia/llama-3.3-nemotron-super-49b-v1"),
    name="dice_agent",
    description=(
        "hello world agent that can roll a dice of 8 sides and check prime"
        " numbers."
    ),
    instruction="""
      You roll dice and answer questions about the outcome of the dice rolls.
    """,
    # tools=[
    #     roll_die,
    #     check_prime,
    # ],
)