from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Optional, List
from datetime import date
from dotenv import load_dotenv
import asyncio
import os

class Airbnb(BaseModel):
    """
    Model representing an Airbnb listing with comprehensive attributes.

    Attributes:
        id (str): Unique identifier for the listing.
        name (str): Title of the listing.
        host_name (str): Name of the host.
        location (str): Location of the listing (city, state, country).
        latitude (float): Latitude coordinate of the listing.
        longitude (float): Longitude coordinate of the listing.
        price_per_night (float): Nightly price in USD.
        min_nights (int): Minimum nights required for booking.
        max_nights (int): Maximum nights allowed for booking.
        availability_365 (int): Number of days the listing is available per year.
        amenities (List[str]): List of amenities offered.
        property_type (str): Type of property (e.g., Apartment, House, Villa).
        room_type (str): Type of room (e.g., Entire home, Private room).
        number_of_guests (int): Max number of guests allowed.
        number_of_bedrooms (int): Total bedrooms.
        number_of_beds (int): Total beds.
        number_of_bathrooms (float): Total bathrooms.
        rating (Optional[float]): Average rating score (0.0 - 5.0).
        number_of_reviews (int): Total number of reviews.
        last_review_date (Optional[date]): Date of the last review.
        is_superhost (bool): Whether the host is a Superhost.
    """
    id: str
    name: str
    host_name: str
    location: str
    latitude: float
    longitude: float
    price_per_night: float
    min_nights: int
    max_nights: int
    availability_365: int
    amenities: List[str]
    property_type: str
    room_type: str
    number_of_guests: int
    number_of_bedrooms: int
    number_of_beds: int
    number_of_bathrooms: float
    rating: Optional[float] = None
    number_of_reviews: int
    last_review_date: Optional[date] = None
    is_superhost: bool

load_dotenv()   

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

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
            tools = await load_mcp_tools(session)
            # structured_llm = llm.with_structured_output(Airbnb)
            agent = create_react_agent(llm, tools)

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can scrape websites, crawl pages, and extract data using Airbnb tools. Think step by step and use the appropriate tools to help the user."
                }
            ]

            print("Available Tools -", *[tool.name for tool in tools])
            print("-" * 60)
            print("🏡 Welcome to Airbnb AI Search!")
            while True:
                user_input = input("\nYou: ").lower()
                if user_input == ["q" ,"bye","quit"]:
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
