from sentence_transformers import SentenceTransformer
import asyncio
import numpy as np

# Load local embedding model
model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

async def generate_embeddings(
    strings: list[str]
):

    loop = asyncio.get_event_loop()

    embeddings = await loop.run_in_executor(
        None,
        lambda: model.encode(strings)
    )

    embeddings = [
        embedding.tolist()
        for embedding in embeddings
    ]

    return embeddings


if __name__ == "__main__":

    texts = [
        "Hello how are you",
        "I like Machine Learning"
    ]

    embeddings = asyncio.run(
        generate_embeddings(texts)
    )

    print(np.array(embeddings).shape)

    print(embeddings)