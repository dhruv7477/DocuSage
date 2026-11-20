"""
scripts/security/export_openapi.py
──────────────────────────────────
Write the API's OpenAPI schema to a file for the ZAP API scan in CI.

The schema is built from the app object, so no server is needed and the
deployed app keeps /docs, /redoc and /openapi.json switched off. Paths that
would end the scan's session (logout) are left out.

    python scripts/security/export_openapi.py --server https://localhost openapi.json

An output of "-" writes the schema to stdout. CI pipes this file into the
API container (`python - ...`, PYTHONPATH=/src): the image has the app and
its dependencies but not this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Calling these mid-scan would revoke the scanner's token.
SESSION_ENDING_PATHS = ("/api/v1/auth/logout",)


def build_schema(server: str, exclude: tuple[str, ...]) -> dict:
    """The app's OpenAPI schema, served from `server`, without `exclude` paths."""
    from src.main import create_app

    schema = create_app().openapi()
    schema["servers"] = [{"url": server}]
    schema["paths"] = {
        path: ops for path, ops in schema["paths"].items() if path not in exclude
    }
    return schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the OpenAPI schema.")
    parser.add_argument("output", help='file to write, or "-" for stdout')
    parser.add_argument("--server", required=True, help="base URL the scan targets")
    args = parser.parse_args(argv)

    schema = build_schema(args.server, SESSION_ENDING_PATHS)
    text = json.dumps(schema, indent=2)
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text, encoding="utf-8")
    print(f"{len(schema['paths'])} paths exported", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
