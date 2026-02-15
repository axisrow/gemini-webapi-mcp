"""MCP Server for Google Gemini via browser cookies."""

from gemini_webapi_mcp.server import mcp


def main():
    """Entry point for the gemini-webapi-mcp command."""
    mcp.run()
