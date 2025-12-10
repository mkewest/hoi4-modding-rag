"""Entry point to start the MCP server."""

from __future__ import annotations

from src.mcp_server.server import create_server


def main() -> None:
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
