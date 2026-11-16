from __future__ import annotations

"""
OctaBroker MCP Client Integration.

Connects to https://mcp.octabroker.com/mcp using JSON-RPC 2.0 over HTTP.
Provides:
- Account synchronization (Equity, Balance, Margin, Leverage)
- Active positions & trade history inspection
- Integration with Institutional Confluence Engine for live precision predictions
"""

import logging
import os
import requests
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

logger = logging.getLogger("OctaMCPClient")

OCTA_MCP_URL = os.getenv("OCTA_MCP_URL", "https://mcp.octabroker.com/mcp")


class OctaMCPClient:
    def __init__(self, token: str | None = None, url: str = OCTA_MCP_URL):
        self.url = url
        self.token = token or os.getenv("OCTA_MCP_TOKEN", "")
        self.request_id = 0

    @property
    def is_configured(self) -> bool:
        return bool(self.token and len(self.token) > 10)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _call_rpc(self, method: str, params: dict | None = None) -> dict:
        if not self.is_configured:
            return {"error": "OCTA_MCP_TOKEN is not configured in .env"}

        self.request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {},
        }
        try:
            resp = requests.post(self.url, headers=self._headers(), json=payload, timeout=12)
            if resp.status_code == 401:
                return {"error": f"Octa MCP Authentication Failed: {resp.text}"}
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("Octa MCP error: %s", exc)
            return {"error": str(exc)}

    def list_tools(self) -> dict:
        """List all available tools provided by OctaBroker MCP server."""
        return self._call_rpc("tools/list")

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        """Call a specific OctaBroker tool (e.g. get_account_summary, get_positions, etc.)."""
        return self._call_rpc("tools/call", {"name": name, "arguments": arguments or {}})

    def get_account_info(self) -> dict:
        """Attempts to retrieve account information from Octa MCP tools."""
        tools_resp = self.list_tools()
        if "error" in tools_resp:
            return tools_resp
        
        # Discover account tool
        tools = tools_resp.get("result", {}).get("tools", [])
        tool_names = [t.get("name") for t in tools]
        
        for candidate in ["get_account", "account_info", "get_accounts", "get_profile", "get_trading_accounts"]:
            if candidate in tool_names:
                return self.call_tool(candidate)
        
        return {"tools_available": tool_names}


if __name__ == "__main__":
    client = OctaMCPClient()
    print("Octa MCP Client configured:", client.is_configured)
    if client.is_configured:
        print("Testing tools/list...")
        print(client.list_tools())
    else:
        print("To connect, add OCTA_MCP_TOKEN=<your_token> to your .env file.")
