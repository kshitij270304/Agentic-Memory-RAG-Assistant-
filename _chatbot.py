"""Backward-compatible entry point for the secure RAG application."""

import asyncio

from mem.response_generator import run_chat


if __name__ == "__main__":
    asyncio.run(run_chat(user_id=1))
