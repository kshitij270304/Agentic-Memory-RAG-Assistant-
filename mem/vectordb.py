from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    VectorParams,
    models,
)

import asyncio

# =========================
# QDRANT CONFIG
# =========================

client = AsyncQdrantClient(
    url="http://localhost:6333"
)

COLLECTION_NAME = "memories"

# IMPORTANT:
# all-MiniLM-L6-v2 = 384 dimensions
EMBEDDING_DIMENSION = 384


# =========================
# DATA MODELS
# =========================

class EmbeddedMemory(BaseModel):

    user_id: int

    memory_text: str

    categories: list[str]

    date: str

    embedding: list[float]


class RetrievedMemory(BaseModel):

    point_id: str

    user_id: int

    memory_text: str

    categories: list[str]

    date: str

    score: float


# =========================
# CREATE COLLECTION
# =========================

async def create_memory_collection():

    exists = await client.collection_exists(
        COLLECTION_NAME
    )

    if not exists:

        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )

        # user_id index
        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="user_id",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

        # category index
        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="categories",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        print("\n✅ Memory collection created\n")

    else:

        print("\n✅ Collection already exists\n")


# =========================
# INSERT MEMORIES
# =========================

async def insert_memories(
    memories: list[EmbeddedMemory]
):

    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=uuid4().hex,
                payload={
                    "user_id": memory.user_id,
                    "categories": memory.categories,
                    "memory_text": memory.memory_text,
                    "date": memory.date,
                },
                vector=memory.embedding,
            )
            for memory in memories
        ],
    )


# =========================
# SEARCH MEMORIES
# =========================

async def search_memories(
    search_vector: list[float],
    user_id: int,
    categories: Optional[list[str]] = None,
):

    must_conditions = [
        models.FieldCondition(
            key="user_id",
            match=models.MatchValue(value=user_id),
        )
    ]

    # optional category filter
    if categories is not None and len(categories) > 0:

        must_conditions.append(
            models.FieldCondition(
                key="categories",
                match=models.MatchAny(any=categories),
            )
        )

    result = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=search_vector,
        with_payload=True,
        query_filter=Filter(must=must_conditions),
        score_threshold=0.3,
        limit=5,
    )

    return [
        convert_retrieved_records(point)
        for point in result.points
        if point is not None
    ]


# =========================
# DELETE ALL USER MEMORIES
# =========================

async def delete_user_records(user_id):

    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(
                            value=user_id
                        ),
                    )
                ]
            )
        ),
    )


# =========================
# DELETE SPECIFIC MEMORIES
# =========================

async def delete_records(point_ids):

    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(
            points=point_ids
        ),
    )


# =========================
# FETCH ALL USER MEMORIES
# =========================

async def fetch_all_user_records(user_id):

    result = await client.query_points(
        collection_name=COLLECTION_NAME,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(
                        value=user_id
                    ),
                )
            ]
        ),
        limit=100,
    )

    return [
        convert_retrieved_records(point)
        for point in result.points
    ]


# =========================
# CONVERT DB RECORD
# =========================

def convert_retrieved_records(point):

    return RetrievedMemory(
        point_id=point.id,
        user_id=point.payload["user_id"],
        memory_text=point.payload["memory_text"],
        categories=point.payload["categories"],
        date=point.payload["date"],
        score=point.score,
    )


# =========================
# GET ALL CATEGORIES
# =========================

async def get_all_categories(user_id):

    facet_filter = Filter(
        must=[
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(
                    value=user_id
                ),
            )
        ]
    )

    facet_result = await client.facet(
        collection_name=COLLECTION_NAME,
        key="categories",
        facet_filter=facet_filter,
        limit=1000,
    )

    unique_categories = [
        hit.value
        for hit in facet_result.hits
    ]

    return unique_categories


# =========================
# STRING FORMATTER
# =========================

def stringify_retrieved_point(
    retrieved_memory: RetrievedMemory
):

    return (
        f"{retrieved_memory.memory_text} "
        f"(Categories: {retrieved_memory.categories}) "
        f"Relevance: {retrieved_memory.score:.2f}"
    )


# =========================
# TEST
# =========================

if __name__ == "__main__":

    asyncio.run(
        create_memory_collection()
    )