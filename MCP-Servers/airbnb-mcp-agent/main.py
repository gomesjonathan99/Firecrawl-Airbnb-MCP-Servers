# importing necessary libraries
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import date
from dotenv import load_dotenv
import asyncio
import os

# loading .env variables
load_dotenv() 

# Creating a Response class 
class AirbnbSearch(BaseModel):
    location: str = Field(..., description="City or location to search for stays")
    check_in: date = Field(..., description="Check-in date")
    check_out: date = Field(..., description="Check-out date")
    guests: int = Field(..., ge=1, description="Number of guests")
    min_price: Optional[float] = Field(None, ge=0, description="Minimum price per night")
    
# initializing the llm
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# server paramerters
server_params = StdioServerParameters(
    command="npx",
    args= [
        "-y",
        "@openbnb/mcp-server-airbnb"
    ]

)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session) # retrieving tools from server
            agent = create_react_agent(llm, tools, response_format=AirbnbSearch) # passing response format calss

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can scrape Airbnb website, crawl pages,"
                                "and extract data using Airbnb tools."
                                "Think step by step and use the appropriate tools to help the user."
                }
            ]

            print("Available Tools -", *[tool.name for tool in tools])
            print("-" * 60)
            print("🏡 Welcome to Airbnb AI Search!")
            while True:
                user_input = input("\nYou: ").lower()
                if user_input in ["q" ,"bye","quit"]:
                    print("Goodbye")
                    break

                messages.append({"role": "user", "content": user_input[:175000]})

                try:
                    agent_response = await agent.ainvoke({"messages": messages})

                    ai_message = agent_response["messages"][-1].content
                    print("\nAgent:", ai_message)
                except Exception as e:
                    print("Error:", e)


if __name__ == "__main__":
    asyncio.run(main())
