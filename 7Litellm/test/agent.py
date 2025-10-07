import asyncio
import uuid
from dotenv import load_dotenv

# Google ADK Imports
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.models.lite_llm import LiteLlm

# OpenAPI Toolset Import
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

# Load environment variables (if needed)
load_dotenv()

# --- Constants ---
APP_NAME = "pokemon_agent_app"
USER_ID = "user_pokemon"
SESSION_ID = f"session_pokemon_{uuid.uuid4()}"
AGENT_NAME = "pokemon_expert_agent"
GEMINI_MODEL = "gemini-2.0-pro"  # or gemini-2.0-flash if faster

# --- PokéAPI OpenAPI Specification (Simplified) ---
pokeapi_spec_string = """
{
  "openapi": "3.0.0",
  "info": {
    "title": "PokéAPI",
    "version": "2.0",
    "description": "An API providing data about Pokémon."
  },
  "servers": [
    { "url": "https://pokeapi.co/api/v2", "description": "Main PokéAPI server" }
  ],
  "paths": {
    "/pokemon/{name}": {
      "get": {
        "summary": "Get Pokémon info",
        "operationId": "getPokemonByName",
        "parameters": [
          {
            "name": "name",
            "in": "path",
            "required": true,
            "description": "Name of the Pokémon",
            "schema": { "type": "string" }
          }
        ],
        "responses": {
          "200": {
            "description": "Pokémon details",
            "content": {
              "application/json": {
                "schema": { "type": "object" }
              }
            }
          },
          "404": { "description": "Pokémon not found" }
        }
      }
    },
    "/type": {
      "get": {
        "summary": "List all Pokémon types",
        "operationId": "listTypes",
        "responses": {
          "200": {
            "description": "List of types",
            "content": {
              "application/json": {
                "schema": { "type": "object" }
              }
            }
          }
        }
      }
    },
    "/ability/{name}": {
      "get": {
        "summary": "Get info about a Pokémon ability",
        "operationId": "getAbility",
        "parameters": [
          {
            "name": "name",
            "in": "path",
            "required": true,
            "description": "Name of the ability",
            "schema": { "type": "string" }
          }
        ],
        "responses": {
          "200": {
            "description": "Ability details",
            "content": {
              "application/json": {
                "schema": { "type": "object" }
              }
            }
          }
        }
      }
    }
  }
}
"""

# --- Create Toolset from PokéAPI Spec ---
pokemon_toolset = OpenAPIToolset(
    spec_str=pokeapi_spec_string,
    spec_str_type='json'
)

# --- Create the Agent ---
root_agent = LlmAgent(
    name=AGENT_NAME,
    model=LiteLlm(model="openai/nvidia/llama-3.3-nemotron-super-49b-v1"),
    tools=[pokemon_toolset],
    instruction="""
You are a Pokémon expert assistant. Use the Pokémon API to retrieve details about Pokémon, their types, and abilities.
When users ask about a Pokémon, retrieve its height, weight, types, abilities, and base stats.
When users ask about types or abilities, provide definitions and examples.
Always include the Pokémon name or type in your response clearly.
""",
    description="Provides Pokémon information using the PokéAPI."
)

# --- Session and Runner Setup ---
async def setup_runner():
    session_service = InMemorySessionService()
    runner = Runner(
        agent=pokemon_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    return runner

# --- Function to Run Queries ---
async def call_pokemon_agent(query, runner):
    print("\n=== Pokémon Agent Interaction ===")
    print(f"Query: {query}")

    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "No response received."

    try:
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=content
        ):
            if event.get_function_calls():
                call = event.get_function_calls()[0]
                print(f"-> Agent called: {call.name} with args {call.args}")
            elif event.get_function_responses():
                response = event.get_function_responses()[0]
                print(f"-> Function response: {response.name}")
            elif event.is_final_response() and event.content and event.content.parts:
                final_response_text = event.content.parts[0].text.strip()

        print(f"\n✅ Final Response: {final_response_text}")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 40)

# --- Run Example Queries ---
async def run_examples():
    runner = await setup_runner()
    await call_pokemon_agent("Tell me about Pikachu.", runner)
    await call_pokemon_agent("What is the ability 'blaze'?", runner)
    await call_pokemon_agent("List all Pokémon types.", runner)

# --- Main ---
if __name__ == "__main__":
    print("🎮 Starting Pokémon Agent...")
    try:
        asyncio.run(run_examples())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            print("⚠️ Cannot use asyncio.run inside a notebook. Use: await run_examples()")
        else:
            raise
    print("🏁 Done.")
