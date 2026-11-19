"""
Minimal OpenAI-compatible LLM server for the end-to-end demo check.

Standard library only, so it runs in a stock python:3.12-slim container.
It lists the model IDs from MOCK_MODEL_IDS (for the app's health probe),
answers JSON-mode requests with a router decision, and answers everything
else with a fixed, cited sentence, streamed or not.
"""

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_IDS = [m for m in os.environ.get("MOCK_MODEL_IDS", "mock").split(",") if m]
ANSWER = "The internal codename for the Q3 relocation project is HELIOTROPE-7741 [1]."
ROUTER_DECISION = json.dumps({"intent": "rag_query", "documents": []})


ROUTER_PROMPT_MARKER = "Classify into exactly one intent"


def _completion_text(body: dict) -> str:
    """Router prompts get a routing decision; everything else gets the answer."""
    is_router = any(
        ROUTER_PROMPT_MARKER in str(m.get("content") or "")
        for m in body.get("messages", [])
    )
    return ROUTER_DECISION if is_router else ANSWER


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.rstrip("/")
        if path == "/v1/models":
            data = [{"id": m, "object": "model", "owned_by": "mock"} for m in MODEL_IDS]
            self._json({"object": "list", "data": data})
        elif (
            path.startswith("/v1/models/")
            and path.removeprefix("/v1/models/") in MODEL_IDS
        ):
            model = path.removeprefix("/v1/models/")
            self._json({"id": model, "object": "model", "owned_by": "mock"})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        text = _completion_text(body)
        if body.get("stream"):
            self._stream(body.get("model", "mock"), text)
        else:
            self._json(
                {
                    "id": "mock-1",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model", "mock"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            )

    def _stream(self, model: str, text: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        pieces = [text[i : i + 12] for i in range(0, len(text), 12)]
        for i, piece in enumerate(pieces + [""]):
            last = i == len(pieces)
            chunk = {
                "id": "mock-1",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {} if last else {"content": piece},
                        "finish_reason": "stop" if last else None,
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _json(self, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: object) -> None:
        print("mock-llm:", fmt % args, flush=True)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
