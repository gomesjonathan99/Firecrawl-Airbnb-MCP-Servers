"""
MCP Client with Firecrawl Integration and Tool Usage Logging
-----------------------------------------------------------
This script connects to an MCP-compatible Firecrawl server via stdio transport
and uses a LangChain ReAct agent (powered by Gemini) to crawl and summarize web pages.
It includes enhanced logging to show which tools are being used.
"""

import asyncio
import os
import sys
import logging
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import StdOutCallbackHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate API key is available
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables")
    sys.exit(1)

# Configure Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    api_key=GOOGLE_API_KEY,
    temperature=0.2,
    top_p=0.95,
)

# Server configuration
SERVER_PATH = os.getenv("MCP_SERVER_PATH", r"C:\Users\MCP\mcp_server.py")
server_params = StdioServerParameters(
    command="python",
    args=[SERVER_PATH], 
    env=None
)

# Custom callback handler to track tool usage
class ToolUsageCallbackHandler(StdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.tool_usage = []
        
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown_tool")
        print(f"\n🔧 Using tool: {tool_name}")
        print(f"   Input: {input_str}")
        self.tool_usage.append({"tool": tool_name, "input": input_str})
        
    def on_tool_end(self, output, **kwargs):
        print(f"   Output: {output[:150]}..." if len(str(output)) > 150 else f"   Output: {output}")
        
    def get_tool_usage_summary(self):
        if not self.tool_usage:
            return "No tools were used."
            
        summary = "\n🧰 Tools Used:\n"
        for i, usage in enumerate(self.tool_usage, 1):
            summary += f"{i}. {usage['tool']} - Input: {usage['input']}\n"
        return summary

async def run_app(user_question: str):
    """
    Run the LangGraph ReAct agent with tools from Firecrawl MCP server.
    
    Args:
        user_question (str): The query or instruction for the agent to process
        
    Returns:
        dict: The agent's response
    """
    logger.info(f"Processing query: {user_question}")
    
    # Initialize tool usage tracker
    tool_tracker = ToolUsageCallbackHandler()
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                logger.info(f"Launching MCP server from: {SERVER_PATH}")
                logger.info("Initializing MCP session")
                await session.initialize()

                logger.info("Loading MCP tools")
                tools = await load_mcp_tools(session)
                
                # Print available tools
                print("\n📋 Available Tools:")
                for i, tool in enumerate(tools, 1):
                    print(f"--- {i}. {tool.name} ---")
                
                agent = create_react_agent(model, tools)

                logger.info("Invoking agent")
                agent_response = await agent.ainvoke(
                    {
                        "messages": [{"role": "user", "content": user_question}]
                    },
                    config={"callbacks": [tool_tracker]}
                )

                # Handle AIMessage objects in response
                if "messages" in agent_response:
                    last_message = agent_response["messages"][-1]
                    if hasattr(last_message, "content"):
                        formatted_response = last_message.content
                    else:
                        # Fall back to string representation if no content attribute
                        formatted_response = str(last_message)
                else:
                    # If no messages field, return the whole response
                    formatted_response = str(agent_response)
                
                print("\n📌 Final Response:\n", formatted_response)
                
                # Print tool usage summary
                print(tool_tracker.get_tool_usage_summary())
                
                return agent_response
                
    except Exception as e:
        logger.error(f"Error during agent execution: {str(e)}", exc_info=True)
        return {"error": str(e)}

async def interactive_mode():
    """Run the application in interactive mode allowing multiple queries"""
    print("🤖 MCP Firecrawl Agent (powered by Gemini)\n")
    print("Type 'exit' or 'quit' to end the session\n")
    
    while True:
        user_input = input("\n🔍 Enter your query: ")
        if user_input.lower() in ('exit', 'quit'):
            print("Goodbye!")
            break
            
        await run_app(user_input)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Firecrawl Agent")
    parser.add_argument("--query", "-q", type=str, help="Run a single query and exit")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.query:
        asyncio.run(run_app(args.query))
    elif args.interactive:
        asyncio.run(interactive_mode())
    else:
        ...
        # # Default example query
        # test_query = "Summarize https://python.langchain.com/docs/guides/"
        # print(f"Running example query: {test_query}")
        # asyncio.run(run_app(test_query))
