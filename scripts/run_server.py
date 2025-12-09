"""Entry point to start the MCP server."""

from __future__ import annotations

from hoi4_rag.mcp_server import create_server


def main():
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
