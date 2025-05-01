from types_aiobotocore_bedrock_runtime.type_defs import ToolConfigurationTypeDef, ToolTypeDef

from aiollm.tools.tool import Tool
from aiollm.tools.tools import Tools


class FromTool:
    @staticmethod
    def from_tool(tool: Tool) -> ToolTypeDef:
        return {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {"json": tool.parameters},
            }
        }

    @staticmethod
    def from_tools(tools: Tools | list[Tool]) -> ToolConfigurationTypeDef:
        return {"tools": [FromTool.from_tool(tool) for tool in tools]}
