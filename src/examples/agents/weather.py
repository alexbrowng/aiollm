from aiollm.agents.agent import Agent
from aiollm.parameters.parameters import Parameters
from examples.tools import get_weather

agent = Agent(
    name="Weather Agent",
    description="A weather agent that can get the weather for a given location.",
    instructions="You are given a location and you need to return the weather for that location.",
    model="openai:gpt-4o-mini",
    parameters=Parameters(temperature=0.2),
)

agent.tool(
    name=get_weather.tool.name,
    description=get_weather.tool.description,
    parameters=get_weather.tool.parameters,
    strict=get_weather.tool.strict,
    handler=get_weather.handler,
)
