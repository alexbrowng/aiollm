from aiollm.tools.tool import Tool

tool = Tool(
    name="get_weather",
    description="Get the weather for a given location.",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location to get the weather for.",
            },
        },
        "required": ["location"],
        "additionalProperties": False,
    },
    strict=True,
)


async def handler(arguments: dict) -> dict:
    return {
        "location": arguments.get("location"),
        "temperature": 20,
        "temperature_unit": "Celsius",
        "humidity": 50,
        "humidity_unit": "%",
        "wind": 10,
        "wind_unit": "km/h",
        "weather": "sunny",
    }
