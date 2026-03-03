#!/usr/bin/env python3
"""
Send one prebuilt prompt file to the Ollama endpoint and measure response time.

Defaults match scripts/public/shared_scripts/generation/generate_answers.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[3]

OLLAMA_URL = "https://chat.fri.uni-lj.si/ollama/api/generate"
OLLAMA_MODEL = "llama3.3:latest"
MAX_LLM_RETRIES = 3

logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    try:
        from dotenv import load_dotenv as _load

        _load(env_path)
    except ImportError:
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _is_retryable_request_error(exc: BaseException) -> bool:
    if requests is not None:
        if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
            return True
        if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
            return exc.response.status_code in (429, 502, 503, 504)
        return False

    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (429, 502, 503, 504)
    return False


def get_api_key() -> str:
    key = (os.getenv("LLAMA_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Missing LLAMA_API_KEY in environment or .env")
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call the LLM once with a prompt file and record response time."
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=(
            REPO_ROOT
            / "output"
            / "workflow_local_10pct_hpc_bge"
            / "generation"
            / "67cf3b6381b1027333000024.txt"
        ),
        help="Path to a prompt text file that already contains [SYSTEM]/[USER] blocks.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Seconds to wait for response (default: 120).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling top_p (default: 1.0).",
    )
    parser.add_argument(
        "--retry-sleep",
        type=int,
        default=5,
        help="Seconds to sleep between retry attempts (default: 5).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def call_llm_raw(
    prompt_text: str,
    api_key: str,
    timeout: int,
    temperature: float,
    top_p: float,
) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "prompt": prompt_text,
        "options": {
            "temperature": float(temperature),
            "top_p": float(top_p),
        },
    }

    if requests is not None:
        r = requests.post(
            OLLAMA_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        return str(data.get("response", ""))

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data.get("response", ""))


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.prompt_path.exists():
        logger.error("Prompt file not found: %s", args.prompt_path)
        return 1

    _load_dotenv()
    api_key = get_api_key()
    prompt_text = args.prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        logger.error("Prompt file is empty: %s", args.prompt_path)
        return 1

    raw_response = ""
    error_message = None
    elapsed_ms = None

    for attempt in range(MAX_LLM_RETRIES):
        try:
            start = time.perf_counter()
            raw_response = call_llm_raw(
                prompt_text=prompt_text,
                api_key=api_key,
                timeout=args.timeout,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            break
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error_message = str(exc)
            if attempt < MAX_LLM_RETRIES - 1 and _is_retryable_request_error(exc):
                logger.warning(
                    "Call failed (attempt %s/%s): %s; retrying in %ss...",
                    attempt + 1,
                    MAX_LLM_RETRIES,
                    exc,
                    args.retry_sleep,
                )
                time.sleep(args.retry_sleep)
            else:
                logger.error(
                    "Call failed after %s attempt(s): %s",
                    attempt + 1,
                    exc,
                )
                break

    result: dict[str, Any] = {
        "prompt_path": str(args.prompt_path),
        "ollama_url": OLLAMA_URL,
        "model": OLLAMA_MODEL,
        "timeout_seconds": args.timeout,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_retries": MAX_LLM_RETRIES,
        "response_time_ms": elapsed_ms,
        "response": raw_response,
        "error": error_message if raw_response == "" else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0 if elapsed_ms is not None else 1


if __name__ == "__main__":
    sys.exit(main())
