#!/usr/bin/env python3
"""Manual test client: call /v1/models and /v1/chat/completions (stream or not)."""
import argparse
import json
import sys
import threading
import time

try:
    import httpx
except ImportError:
    print("Error: httpx is required. Run: pip install httpx", file=sys.stderr)
    sys.exit(1)

WAIT_MESSAGE_SECONDS = 2.0


def main():
    p = argparse.ArgumentParser(description="Test Cursor CLI to OpenAI API server.")
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="Base URL of the API (default: http://127.0.0.1:8080)",
    )
    p.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="Use streaming chat completion (default: True)",
    )
    p.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Use non-streaming chat completion",
    )
    p.add_argument(
        "--prompt",
        default="Say hello in one sentence.",
        help="User prompt for chat completion",
    )
    args = p.parse_args()

    base = args.base_url.rstrip("/")
    models_url = f"{base}/v1/models"
    chat_url = f"{base}/v1/chat/completions"

    try:
        with httpx.Client(timeout=30.0) as client:
            # 1) GET /v1/models
            print("GET", models_url)
            r = client.get(models_url)
            r.raise_for_status()
            data = r.json()
            print("Models:", json.dumps(data, indent=2, ensure_ascii=False))
            print()

            # 2) POST /v1/chat/completions
            body = {
                "model": "cursor-agent",
                "messages": [{"role": "user", "content": args.prompt}],
                "stream": args.stream,
            }
            print("POST", chat_url, "(stream=%s)" % args.stream)
            if args.stream:
                with client.stream(
                    "POST",
                    chat_url,
                    json=body,
                    timeout=60.0,
                ) as resp:
                    resp.raise_for_status()
                    print("Content:")
                    buffer = b""
                    first_data_received = threading.Event()
                    wait_message_shown = [False]  # list to allow closure to mutate

                    def show_wait_message():
                        time.sleep(WAIT_MESSAGE_SECONDS)
                        if not first_data_received.is_set() and not wait_message_shown[0]:
                            wait_message_shown[0] = True
                            print("Waiting for response...", file=sys.stderr)

                    t = threading.Thread(target=show_wait_message, daemon=True)
                    t.start()

                    done = False
                    for raw in resp.iter_bytes(chunk_size=1024):
                        if done:
                            break
                        if not first_data_received.is_set():
                            first_data_received.set()
                        buffer += raw
                        while b"\n\n" in buffer and not done:
                            event_block, buffer = buffer.split(b"\n\n", 1)
                            for line in event_block.split(b"\n"):
                                line = line.decode("utf-8", errors="replace").strip()
                                if not line.startswith("data: "):
                                    continue
                                payload = line[6:].strip()
                                if payload == "[DONE]":
                                    done = True
                                    break
                                try:
                                    chunk = json.loads(payload)
                                except json.JSONDecodeError:
                                    continue
                                if chunk.get("object") == "error":
                                    err = chunk.get("error", {})
                                    print(err.get("message", str(chunk)), file=sys.stderr)
                                    continue
                                for choice in chunk.get("choices", []):
                                    delta = choice.get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        print(content, end="", flush=True)
                            if done:
                                break
                    print()
            else:
                r = client.post(chat_url, json=body)
                r.raise_for_status()
                data = r.json()
                content = (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                print("Content:", content)
    except httpx.ConnectError as e:
        print("Connection error:", e, file=sys.stderr)
        print("Make sure the server is running (e.g. uvicorn src.main:app --host 127.0.0.1 --port 8080)", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        try:
            # Streaming response body is not read by default; read it for error detail
            body = e.response.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        try:
            err_json = json.loads(body) if body.strip().startswith("{") else {}
            detail = err_json.get("detail", body or str(e))
        except json.JSONDecodeError:
            detail = body or str(e)
        print("HTTP error:", e.response.status_code, detail, file=sys.stderr)
        sys.exit(1)
    except httpx.RemoteProtocolError as e:
        print("Stream error: server closed the connection before completing the response.", file=sys.stderr)
        print("The server may have hit an error (e.g. Cursor agent not found). Check server logs.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
