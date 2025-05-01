from aiollm.tools.tools import Tools
from examples.tools import get_weather

tools = Tools()

tools.register(get_weather.tool, get_weather.handler)
