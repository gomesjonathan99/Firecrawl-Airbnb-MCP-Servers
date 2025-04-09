"""
MCP Server with Firecrawl Integration
------------------------------------
This script sets up an MCP server that provides web scraping and data extraction tools
using the Firecrawl API. It extracts structured data from websites like company information and documentation.
"""

import os
import logging
import sys
from mcp.server.fastmcp import FastMCP
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Schemas
class CompanyInfoSchema(BaseModel):
    company_name: str = Field(description="The name of the company")
    company_mission: str = Field(description="The mission statement or purpose of the company")
    supports_sso: bool = Field(description="Whether the product/service supports Single Sign-On (SSO)")
    is_open_source: bool = Field(description="Whether the product/service is open-source")
    is_in_yc: bool = Field(description="Whether the company is a Y Combinator startup")
    pricing_model: str = Field(description="The pricing model (free, freemium, subscription, etc.)")

class DocumentationSchema(BaseModel):
    title: str = Field(description="The title of the documentation page")
    summary: str = Field(description="A concise summary of the documentation content")
    key_features: List[str] = Field(description="List of key features mentioned")
    code_examples: bool = Field(description="Whether the page contains code examples")

class GeneralContentSchema(BaseModel):
    title: str = Field(description="The title of the page")
    main_topics: List[str] = Field(description="Main topics covered on the page")
    summary: str = Field(description="A brief summary of the page content")

# Firecrawl client wrapper
class FirecrawlClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FIRE_CRAWL_API_KEY")
        if not self.api_key:
            logger.error("FIRE_CRAWL_API_KEY not found")
            raise ValueError("Missing API key")
        self.app = FirecrawlApp(api_key=self.api_key)

    def extract_structured_data(self, url: str, schema_type: str = "company") -> Dict[str, Any]:
        if not url.startswith("http"):
            return {"error": "Invalid URL"}

        # Schema switch
        if schema_type == "company":
            schema = CompanyInfoSchema.model_json_schema()
        elif schema_type == "documentation":
            schema = DocumentationSchema.model_json_schema()
        else:
            schema = GeneralContentSchema.model_json_schema()

        try:
            response = self.app.scrape_url(url, {
                "formats": ["json", "markdown"],
                "jsonOptions": {"schema": schema}
            })

            if response.get("json"):
                return response["json"]
            elif response.get("markdown"):
                return {
                    "raw_text": response["markdown"],
                    "note": "Structured data not found. Returned raw markdown content."
                }
            else:
                return {"error": "No data extracted from the page."}

        except Exception as e:
            return {"error": f"Extraction failed: {str(e)}"}

    def get_website_content(self, url: str) -> Dict[str, Any]:
        if not url or not url.startswith("http"):
            return {"error": "Invalid URL. Must start with http:// or https://"}
        try:
            response = self.app.scrape_url(url, {"formats": ["markdown"]})
            if response and response.get("markdown"):
                return {"content": response["markdown"]}
            else:
                return {"error": "No content could be extracted"}
        except Exception as e:
            return {"error": f"Error fetching content: {str(e)}"}

# MCP Server setup
def setup_mcp_server():
    logger.info("Setting up MCP server")
    mcp = FastMCP("Firecrawl")

    try:
        firecrawl_client = FirecrawlClient()
    except ValueError as e:
        logger.error(f"Failed to initialize Firecrawl client: {str(e)}")
        return None

    # Tool 1: Extract documentation data
    def extract_documentation(url: str) -> Dict[str, Any]:
        if not url:
            return {"error": "URL cannot be empty"}
        return firecrawl_client.extract_structured_data(url, schema_type="documentation")

    # Tool 2: Extract general content summary
    def summarize_webpage(url: str) -> Dict[str, Any]:
        if not url:
            return {"error": "URL cannot be empty"}
        return firecrawl_client.get_website_content(url)

    # Tool 3: Extract company information
    def extract_company_info(url: str) -> Dict[str, Any]:
        if not url:
            return {"error": "URL cannot be empty"}
        return firecrawl_client.extract_structured_data(url, schema_type="company")

    # Register tools
    mcp.add_tool(extract_documentation) # ok
    mcp.add_tool(summarize_webpage) # good
    mcp.add_tool(extract_company_info)

    return mcp

if __name__ == "__main__":
    mcp = setup_mcp_server()
    if mcp:
        try:
            logger.info("--- Launching MCP ---")
            mcp.run(transport="stdio")
        except Exception as e:
            logger.error(f"Error running MCP server: {str(e)}", exc_info=True)
            sys.exit(1)
    else:
        logger.error("Failed to set up MCP server, exiting")
        sys.exit(1)
