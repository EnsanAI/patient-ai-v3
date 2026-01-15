import asyncio
import json
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mcp-tester")

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
except ImportError:
    logger.error("❌ MCP package not found. Please run: pip install mcp")
    sys.exit(1)

# Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")

async def test_mcp_server():
    logger.info("="*60)
    logger.info(f"🧪 MCP SERVER DIAGNOSTIC TOOL")
    logger.info(f"Target URL: {MCP_SERVER_URL}")
    logger.info("="*60)

    try:
        logger.info(f"🔌 Connecting to {MCP_SERVER_URL}...")
        
        async with sse_client(MCP_SERVER_URL) as (read, write):
            async with ClientSession(read, write) as session:
                logger.info("✅ Connection established!")
                
                # Initialize
                await session.initialize()
                logger.info("✅ Session initialized")
                
                # 1. List Tools
                logger.info("\n📋 DISCOVERING TOOLS...")
                result = await session.list_tools()
                
                if not result or not hasattr(result, 'tools'):
                    logger.error("❌ Failed to list tools (empty response)")
                    return

                tools = result.tools
                logger.info(f"Found {len(tools)} tools:")
                
                tool_names = []
                for t in tools:
                    logger.info(f"  - {t.name}: {t.description[:60] if t.description else 'No description'}...")
                    tool_names.append(t.name)
                
                # 2. Test Execution
                logger.info("\n🚀 EXECUTING TEST CALLS...")
                
                # Define test scenarios (Intent -> Potential Tool Names)
                scenarios = [
                    {
                        "name": "List Doctors",
                        "candidates": ["get_doctors", "list_doctors", "doctors_list"],
                        "args": {}
                    },
                    {
                        "name": "Clinic Info",
                        "candidates": ["get_clinic_info", "clinic_info", "get_clinic_details"],
                        "args": {}
                    },
                    {
                        "name": "List Procedures",
                        "candidates": ["get_all_dental_procedures", "list_procedures", "get_procedures"],
                        "args": {}
                    },
                    {
                        "name": "System Health",
                        "candidates": ["check_system_health", "health_check", "ping"],
                        "args": {}
                    }
                ]
                
                for scenario in scenarios:
                    logger.info(f"\nTesting: {scenario['name']}")
                    
                    # Find matching tool
                    tool_to_call = None
                    for candidate in scenario['candidates']:
                        if candidate in tool_names:
                            tool_to_call = candidate
                            break
                    
                    if tool_to_call:
                        logger.info(f"  → Calling tool: '{tool_to_call}'")
                        try:
                            # Call the tool
                            result = await session.call_tool(tool_to_call, scenario['args'])
                            
                            # Output result
                            if result.content:
                                text = result.content[0].text
                                logger.info(f"  ✅ Success! Response length: {len(text)} chars")
                            else:
                                logger.info("  ✅ Success! (Empty response)")
                                
                        except Exception as e:
                            logger.error(f"  ❌ Failed: {e}")
                    else:
                        logger.warning(f"  ⚠️ No matching tool found for {scenario['name']}")

    except Exception as e:
        logger.error(f"\n❌ CRITICAL ERROR: {e}")
        logger.error("Ensure the MCP server is running and accessible.")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())