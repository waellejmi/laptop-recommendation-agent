import json
import logging
from pathlib import Path

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    TextContent,
    Tool,
)
from pydantic import FileUrl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Server("mcp-data")


@app.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri=FileUrl("file:///data/laptops_enhanced.csv"),
            name="Enhanced Laptop Dataset",
            mimeType="text/csv",
            description="Laptop inventory with normalized specs and scores",
        ),
        Resource(
            uri=FileUrl("file:///data/games-system-requirements/game_db.csv"),
            name="Game Requirements Database",
            mimeType="text/csv",
            description="80K+ games with system requirements (CPU, GPU, RAM)",
        ),
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    uri_str = str(uri)

    uri_mapping = {
        "file:///data/games-system-requirements/game_db.csv": "./mcp_data/game_db.csv",
        "file:///data/laptops_enhanced.csv": "./mcp_data/laptops_enhanced.csv",
    }

    if uri_str not in uri_mapping:
        logger.error(f"Failed to find URI: '{uri_str}'")
        raise ValueError(f"Unknown resource URI: {uri_str}")

    file_path = uri_mapping[uri_str]

    return json.dumps(
        {
            "uri": uri_str,
            "path": file_path,
            "exists": Path(file_path).exists(),
        }
    )


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "online_lookup":
        raise ValueError(f"Unknown tool: {name}")

    game_name = arguments.get("game_name")
    if not game_name:
        raise ValueError("Missing 'game_name' argument")

    from sys_req_lookup_tool import online_lookup as perform_lookup

    try:
        online_result = perform_lookup(game_name)
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "success": True,
                        "data": {
                            "game_name": online_result["game_name"],
                            "cpu": online_result["cpu"],
                            "gpu": online_result["gpu"],
                            "ram": online_result["ram"],
                        },
                    }
                ),
            )
        ]
    except Exception as e:
        return [
            TextContent(
                type="text", text=json.dumps({"success": False, "error": str(e)})
            )
        ]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="online_lookup",
            description="Lookup system requirements for a game online from SystemRequirementsLab.",
            inputSchema={
                "type": "object",
                "required": ["game_name"],
                "properties": {
                    "game_name": {
                        "type": "string",
                        "description": "The name of the game to look up.",
                    },
                },
            },
        )
    ]


async def arun():
    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(arun())
