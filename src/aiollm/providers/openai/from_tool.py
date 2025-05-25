from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from aiollm.tools.tool import Tool
from aiollm.tools.tools import Tools


class FromTool:
    @staticmethod
    def from_tool(tool: Tool) -> ChatCompletionToolParam:
        return ChatCompletionToolParam(
            type="function",
            function=FunctionDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters.to_primitives(),
                strict=tool.strict,
            ),
        )

    @staticmethod
    def from_tools(tools: Tools | list[Tool]) -> list[ChatCompletionToolParam]:
        return [FromTool.from_tool(tool) for tool in tools]
