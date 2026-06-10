import asyncio
import ollama

from mem.generate_embeddings import generate_embeddings
from mem.vectordb import (
    search_memories,
    insert_memories,
    EmbeddedMemory,
)

from datetime import datetime

# =========================
# USER CONFIG
# =========================

USER_ID = 1

# short-term conversation history
messages = []

# =========================
# STORE MEMORY
# =========================

async def store_memory(user_message: str):

    # generate embedding
    embedding = (
        await generate_embeddings([user_message])
    )[0]

    # store memory
    await insert_memories(
        [
            EmbeddedMemory(
                user_id=USER_ID,
                memory_text=user_message,
                categories=["general"],
                date=datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                ),
                embedding=embedding,
            )
        ]
    )

# =========================
# SEARCH MEMORIES
# =========================

async def retrieve_memories(query: str):

    embedding = (
        await generate_embeddings([query])
    )[0]

    memories = await search_memories(
        search_vector=embedding,
        user_id=USER_ID,
    )

    return memories

# =========================
# MAIN CHAT LOOP
# =========================

async def main():

    print("\n Local Memory Agent Started!\n")

    while True:

        user_input = input("User: ")

        if user_input.lower() in ["exit", "quit"]:

            print("\nGoodbye 👋")
            break

        # add user message
        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        # =========================
        # MEMORY RETRIEVAL
        # =========================

        related_memories = await retrieve_memories(
            user_input
        )

        related_memories_text = "\n- ".join(
            [
                memory.memory_text
                for memory in related_memories
            ]
        )

        print("\n[Retrieved Memories]")
        print(related_memories_text)

        # =========================
        # SYSTEM PROMPT
        # =========================

        system_prompt = f"""
You are a helpful AI assistant with long-term memory.

Relevant memories about the user:

{related_memories_text}

Use these memories only if relevant.
Be conversational and concise.
"""

        # =========================
        # OLLAMA RESPONSE
        # =========================

        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ]
            + messages[-6:]  # recent context only
        )

        answer = response["message"]["content"]

        print(f"\nAssistant: {answer}\n")

        # add assistant response
        messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        # =========================
        # STORE NEW MEMORY
        # =========================

        await store_memory(user_input)


# =========================
# START APP
# =========================

if __name__ == "__main__":

    asyncio.run(main())