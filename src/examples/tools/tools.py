from aiollm.json_schema.object import Object
from aiollm.json_schema.property import Property
from aiollm.json_schema.string import String
from aiollm.tools.tools import Tools

tools = Tools()


@tools.register(
    Object(
        Property("location", String(description="The location to get the weather for.")),
    )
)
async def get_weather(arguments: dict, _context: dict) -> dict:
    """Get the weather for a given location."""
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
