"""Entry point to start the MCP server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from mcp_server.server import create_server  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="HOI4 RAG MCP server")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Run a quick startup/status check and exit (does not start the server).",
    )
    args = parser.parse_args(argv)

    if args.status:
        print("Status: ready to start. (Status check only; server not started.)", flush=True)
        return

    server = create_server()
    try:
        server.run()
    except KeyboardInterrupt:
        # Avoid writing to stdout to keep MCP transport clean
        sys.stderr.write("MCP server stopped by user.\n")


if __name__ == "__main__":
    main()
