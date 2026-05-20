import dspy

from pydantic import BaseModel
from datetime import datetime

from mem.generate_embeddings import generate_embeddings

from mem.vectordb import (
    EmbeddedMemory,
    RetrievedMemory,
    delete_records,
    insert_memories,
    search_memories,
)

# =========================
# DSPY CONFIG
# =========================

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

# =========================
# LOCAL OLLAMA MODEL
# =========================

local_lm = dspy.LM(
    model="ollama_chat/llama3",
    api_base="http://localhost:11434",
    temperature=0.3,
    max_tokens=2048,
)

# =========================
# MEMORY DATA MODEL
# =========================

class MemoryWithIds(BaseModel):

    memory_id: int

    memory_text: str

    memory_categories: list[str]


# =========================
# MEMORY UPDATE SIGNATURE
# =========================

class UpdateMemorySignature(dspy.Signature):
    """
    Decide how memories should be updated.

    Actions:
    - ADD
    - UPDATE
    - DELETE
    - NOOP

    Keep memories:
    - short
    - atomic
    - factual
    """

    messages: list[dict] = dspy.InputField()

    existing_memories: list[MemoryWithIds] = dspy.InputField()

    summary: str = dspy.OutputField(
        description="Very short summary"
    )


# =========================
# CATEGORY NORMALIZER
# =========================

def normalize_categories(categories):

    # string -> list
    if isinstance(categories, str):

        categories = [categories]

    # null protection
    if categories is None:

        categories = ["general"]

    # invalid structure
    if not isinstance(categories, list):

        categories = ["general"]

    # ensure strings only
    cleaned_categories = []

    for category in categories:

        cleaned_categories.append(
            str(category).strip().lower()
        )

    # empty fallback
    if len(cleaned_categories) == 0:

        cleaned_categories = ["general"]

    return cleaned_categories


# =========================
# MEMORY AGENT
# =========================

async def update_memories_agent(
    user_id: int,
    messages: list[dict],
    existing_memories: list[RetrievedMemory],
):

    # -------------------------
    # HELPER
    # -------------------------

    def get_point_id_from_memory_id(memory_id):

        return existing_memories[
            memory_id
        ].point_id

    # -------------------------
    # ADD MEMORY
    # -------------------------

    async def add_memory(
        memory_text: str,
        categories,
    ) -> str:

        categories = normalize_categories(
            categories
        )

        print("\n[ADD MEMORY]")
        print(memory_text)
        print("Categories:", categories)

        embeddings = await generate_embeddings(
            [memory_text]
        )

        await insert_memories(
            memories=[
                EmbeddedMemory(
                    user_id=user_id,
                    memory_text=memory_text,
                    categories=categories,
                    date=datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    embedding=embeddings[0],
                )
            ]
        )

        return (
            f"Added memory: {memory_text}"
        )

    # -------------------------
    # UPDATE MEMORY
    # -------------------------

    async def update(
        memory_id: int,
        updated_memory_text: str,
        categories,
    ):

        categories = normalize_categories(
            categories
        )

        print("\n[UPDATE MEMORY]")
        print(
            "OLD:",
            existing_memories[
                memory_id
            ].memory_text,
        )
        print(
            "NEW:",
            updated_memory_text
        )

        point_id = get_point_id_from_memory_id(
            memory_id
        )

        # remove old memory
        await delete_records([point_id])

        # generate embedding
        embeddings = await generate_embeddings(
            [updated_memory_text]
        )

        # insert updated memory
        await insert_memories(
            memories=[
                EmbeddedMemory(
                    user_id=user_id,
                    memory_text=updated_memory_text,
                    categories=categories,
                    date=datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    embedding=embeddings[0],
                )
            ]
        )

        return (
            f"Updated memory {memory_id}"
        )

    # -------------------------
    # DELETE MEMORY
    # -------------------------

    async def delete(
        memory_ids: list[int]
    ):

        print("\n[DELETE MEMORY]")

        point_ids = []

        for memory_id in memory_ids:

            print(
                existing_memories[
                    memory_id
                ].memory_text
            )

            point_ids.append(
                get_point_id_from_memory_id(
                    memory_id
                )
            )

        await delete_records(point_ids)

        return (
            f"Deleted memories "
            f"{memory_ids}"
        )

    # -------------------------
    # NO OPERATION
    # -------------------------

    async def noop():

        print(
            "\n[NO MEMORY ACTION]"
        )

        return "No changes required"

    # =========================
    # DSPY REACT AGENT
    # =========================

    memory_updater = dspy.ReAct(
        UpdateMemorySignature,
        tools=[
            add_memory,
            update,
            delete,
            noop,
        ],
        max_iters=3,
    )

    # indexed memory mapping
    memory_ids = [
        MemoryWithIds(
            memory_id=idx,
            memory_text=m.memory_text,
            memory_categories=m.categories,
        )
        for idx, m in enumerate(
            existing_memories
        )
    ]

    # run local llama3
    with dspy.context(lm=local_lm):

        out = await memory_updater.acall(
            messages=messages,
            existing_memories=memory_ids,
        )

    return out.summary


# =========================
# MAIN MEMORY UPDATE
# =========================

async def update_memories(
    user_id: int,
    messages: list[dict],
):

    latest_user_message = [
        x["content"]
        for x in messages
        if x["role"] == "user"
    ][-1]

    # generate embedding
    embedding = (
        await generate_embeddings(
            [latest_user_message]
        )
    )[0]

    # retrieve related memories
    retrieved_memories = (
        await search_memories(
            search_vector=embedding,
            user_id=user_id,
        )
    )

    # run memory agent
    response = (
        await update_memories_agent(
            user_id=user_id,
            existing_memories=retrieved_memories,
            messages=messages,
        )
    )

    return response


# =========================
# TEST
# =========================

async def test():

    messages = [
        {
            "role": "user",
            "content":
            "My favorite city is Tokyo"
        }
    ]

    response = await update_memories(
        user_id=1,
        messages=messages,
    )

    print("\nSUMMARY:")
    print(response)


if __name__ == "__main__":

    import asyncio

    asyncio.run(test())