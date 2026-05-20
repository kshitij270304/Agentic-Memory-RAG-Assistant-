import dspy
import asyncio

from rich.console import Console
from rich.rule import Rule

import warnings

warnings.filterwarnings("ignore")

from mem.generate_embeddings import (
    generate_embeddings,
)

from mem.update_memory import (
    update_memories,
)

from mem.vectordb import (
    search_memories,
)

console = Console(log_path=False)

# =========================
# DISABLE DSPY CACHE
# =========================

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

# =========================
# LOCAL MODEL
# =========================

model = dspy.LM(
    model="ollama_chat/llama3",
    api_base="http://localhost:11434",
    temperature=0.0,
    max_tokens=128,
)

# =========================
# MEMORY RESPONSE MODEL
# =========================

class MemoryResponder(dspy.Signature):
    """
    Answer ONLY using retrieved memories.

    RULES:

    - Use ONLY retrieved memories.
    - Never hallucinate.
    - Never guess.
    - Never add extra conversation.
    - Never ask follow-up questions.
    - Keep answers short.

    If memory does not exist,
    respond EXACTLY with:

    "That information does not exist in memory."
    """

    retrieved_memories: list[str] = dspy.InputField()

    question: str = dspy.InputField()

    response: str = dspy.OutputField()

# =========================
# QUESTION DETECTOR
# =========================

def is_question(text: str):

    text = text.lower().strip()

    question_words = [
        "what",
        "who",
        "where",
        "when",
        "why",
        "how",
        "do",
        "does",
        "did",
        "can",
        "could",
        "is",
        "are",
        "am",
    ]

    if text.endswith("?"):

        return True

    for word in question_words:

        if text.startswith(word + " "):

            return True

    return False

# =========================
# MAIN CHAT LOOP
# =========================

async def run_chat(user_id):

    console.print(
        "\n🧠 Memory System Started!\n",
        style="bold green",
    )

    while True:

        question = console.input(
            "[bold cyan]> [/bold cyan]"
        )

        if question.lower() in [
            "exit",
            "quit",
        ]:

            console.print(
                "\nGoodbye 👋\n",
                style="bold red",
            )

            break

        console.print(
            Rule(style="grey50")
        )

        with console.status(
            "[bold green]Working..."
        ):

            # =========================
            # GENERATE EMBEDDING
            # =========================

            embedding = (
                await generate_embeddings(
                    [question]
                )
            )[0]

            # =========================
            # SEARCH MEMORIES
            # =========================

            retrieved = await search_memories(
                search_vector=embedding,
                user_id=user_id,
            )

            retrieved_memories = [
                memory.memory_text
                for memory in retrieved
            ]

            # =========================
            # QUESTION MODE
            # =========================

            if is_question(question):

                # no memories found
                if len(retrieved_memories) == 0:

                    response = (
                        "That information "
                        "does not exist "
                        "in memory."
                    )

                else:

                    responder = dspy.Predict(
                        MemoryResponder
                    )

                    with dspy.context(lm=model):

                        out = responder(
                            retrieved_memories=retrieved_memories,
                            question=question,
                        )

                    response = out.response

            # =========================
            # MEMORY STORAGE MODE
            # =========================

            else:

                try:

                    # ONLY USER MESSAGE
                    await update_memories(
                        user_id=user_id,
                        messages=[
                            {
                                "role": "user",
                                "content": question,
                            }
                        ],
                    )

                    response = "Stored."

                except Exception as e:

                    response = (
                        f"Memory Error: {e}"
                    )

        console.print(
            f"\n[bold green]AI:[/bold green] "
            f"{response}\n"
        )

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":

    USER_ID = "kshitij"

    asyncio.run(
        run_chat(USER_ID)
    )