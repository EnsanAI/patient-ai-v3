import os
import logging
import time
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client

logger = logging.getLogger("mcp_bridge")

class McpConnection:
    """
    Singleton connector for the CareBot MCP Server.
    Handles the Async SSE connection lifecycle.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(McpConnection, cls).__new__(cls)
            # Default to Docker service URL, fallback to localhost for testing
            cls._instance.url = os.environ.get("MCP_SERVER_URL", "http://localhost:8000/sse")
        return cls._instance

    async def execute(self, tool_name: str, arguments: dict = None) -> str:
        """
        Connects to MCP, executes the tool, and returns the text result.
        """
        if arguments is None: arguments = {}
        
        start_time = time.perf_counter()
        logger.info(f"🔌 [MCP Orchestration] Executing Tool: {tool_name} | URL: {self.url}")
        try:
            # We open a fresh connection for each request to ensure stability
            async with sse_client(self.url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Execute the tool
                    result = await session.call_tool(tool_name, arguments)
                    
                    latency = (time.perf_counter() - start_time) * 1000
                    logger.info(f"⚡ [MCP Latency] {tool_name} responded in {latency:.2f}ms")
                    
                    if not result.content:
                        return None
                        
                    # Return the raw text content for the Agent/Client to parse
                    return result.content[0].text
                    
        except Exception as e:
            logger.error(f"❌ MCP Failure [{tool_name}]: {e}")
            raise e # Re-raise so the calling function knows it failed

    async def read_resource(self, uri: str) -> str:
        """
        Connects to MCP, reads a resource, and returns the text content.
        Useful for fetching lists (doctors, appointments) if no tool is available.
        """
        logger.info(f"🔌 MCP Resource: {uri} | URL: {self.url}")
        try:
            async with sse_client(self.url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.read_resource(uri)
                    if result.contents:
                        return result.contents[0].text
                    return None
        except Exception as e:
            logger.error(f"❌ MCP Resource Failure [{uri}]: {e}")
            return None

    async def list_tools(self):
        """List available tools on the MCP server."""
        try:
            async with sse_client(self.url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    return result
        except Exception as e:
            logger.error(f"❌ MCP List Tools Failure: {e}")
            return None

# Export a global instance
mcp = McpConnection()