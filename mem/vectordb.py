"""Chroma Cloud storage adapter for long-term memories."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel

COLLECTION_NAME = "memories"
RELEVANCE_THRESHOLD = 0.3
SEARCH_LIMIT = 5
CATEGORY_SEARCH_LIMIT = 50

def _load_local_env():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(
        encoding="utf-8"
    ).splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or "=" not in stripped
        ):
            continue

        key, value = stripped.split("=", 1)
        os.environ.setdefault(
            key.strip(),
            value.strip().strip("\"'"),
        )


_load_local_env()

_client = None
_collection = None


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


def _get_client():
    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")

    missing = [
        name
        for name, value in {
            "CHROMA_API_KEY": api_key,
            "CHROMA_TENANT": tenant,
            "CHROMA_DATABASE": database,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing Chroma Cloud configuration: "
            + ", ".join(missing)
        )

    import chromadb

    _client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database,
    )
    return _client


def _get_collection():
    global _collection

    if _collection is None:
        _collection = _get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
            embedding_function=None,
        )

    return _collection


def _normalize_categories(categories: Any) -> list[str]:
    if not isinstance(categories, list):
        return ["general"]

    cleaned = [
        str(category).strip().lower()
        for category in categories
        if str(category).strip()
    ]
    return cleaned or ["general"]


def _build_where_filter(
    user_id: int,
) -> dict[str, Any]:
    return {
        "user_id": user_id,
    }


def _serialize_categories(
    categories: list[str],
) -> str:
    return json.dumps(
        _normalize_categories(categories),
        separators=(",", ":"),
    )


def _deserialize_categories(
    value: Any,
) -> list[str]:
    if isinstance(value, list):
        return _normalize_categories(value)

    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = [
                category.strip()
                for category in value.split(",")
                if category.strip()
            ]
        return _normalize_categories(decoded)

    return ["general"]


def _metadata_to_memory(
    point_id: str,
    document: Optional[str],
    metadata: Optional[dict[str, Any]],
    distance: Optional[float] = None,
) -> RetrievedMemory:
    metadata = metadata or {}
    score = (
        max(-1.0, min(1.0, 1.0 - float(distance)))
        if distance is not None
        else 0.0
    )

    return RetrievedMemory(
        point_id=str(point_id),
        user_id=int(metadata["user_id"]),
        memory_text=document or str(
            metadata.get("memory_text", "")
        ),
        categories=_deserialize_categories(
            metadata.get("categories")
        ),
        date=str(metadata.get("date", "")),
        score=score,
    )


async def create_memory_collection():
    await asyncio.to_thread(_get_collection)
    print("\nChroma memory collection is ready\n")


async def insert_memories(
    memories: list[EmbeddedMemory],
):
    if not memories:
        return

    collection = await asyncio.to_thread(
        _get_collection
    )
    await asyncio.to_thread(
        collection.upsert,
        ids=[
            uuid4().hex
            for _ in memories
        ],
        embeddings=[
            memory.embedding
            for memory in memories
        ],
        documents=[
            memory.memory_text
            for memory in memories
        ],
        metadatas=[
            {
                "user_id": memory.user_id,
                # Chroma metadata values are kept scalar for Cloud and local
                # compatibility. Categories are decoded after retrieval.
                "categories": _serialize_categories(
                    memory.categories
                ),
                "date": memory.date,
            }
            for memory in memories
        ],
    )


async def search_memories(
    search_vector: list[float],
    user_id: int,
    categories: Optional[list[str]] = None,
):
    collection = await asyncio.to_thread(
        _get_collection
    )
    result = await asyncio.to_thread(
        collection.query,
        query_embeddings=[search_vector],
        n_results=(
            CATEGORY_SEARCH_LIMIT
            if categories
            else SEARCH_LIMIT
        ),
        where=_build_where_filter(
            user_id=user_id,
        ),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    memories = [
        _metadata_to_memory(
            point_id=point_id,
            document=document,
            metadata=metadata,
            distance=distance,
        )
        for point_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        )
    ]

    cleaned_categories = set(
        _normalize_categories(categories)
    ) if categories else set()

    filtered = [
        memory
        for memory in memories
        if memory.score >= RELEVANCE_THRESHOLD
        and (
            not cleaned_categories
            or cleaned_categories.intersection(
                memory.categories
            )
        )
    ]
    return filtered[:SEARCH_LIMIT]


async def delete_user_records(user_id: int):
    collection = await asyncio.to_thread(
        _get_collection
    )
    await asyncio.to_thread(
        collection.delete,
        where={
            "user_id": user_id,
        },
    )


async def delete_records(point_ids: list[str]):
    if not point_ids:
        return

    collection = await asyncio.to_thread(
        _get_collection
    )
    await asyncio.to_thread(
        collection.delete,
        ids=[
            str(point_id)
            for point_id in point_ids
        ],
    )


async def fetch_all_user_records(
    user_id: int,
):
    collection = await asyncio.to_thread(
        _get_collection
    )
    result = await asyncio.to_thread(
        collection.get,
        where={
            "user_id": user_id,
        },
        limit=100,
        include=[
            "documents",
            "metadatas",
        ],
    )

    return [
        _metadata_to_memory(
            point_id=point_id,
            document=document,
            metadata=metadata,
        )
        for point_id, document, metadata in zip(
            result.get("ids") or [],
            result.get("documents") or [],
            result.get("metadatas") or [],
        )
    ]


async def get_all_categories(
    user_id: int,
):
    memories = await fetch_all_user_records(
        user_id
    )
    return sorted({
        category
        for memory in memories
        for category in memory.categories
    })


def stringify_retrieved_point(
    retrieved_memory: RetrievedMemory,
):
    return (
        f"{retrieved_memory.memory_text} "
        f"(Categories: {retrieved_memory.categories}) "
        f"Relevance: {retrieved_memory.score:.2f}"
    )


if __name__ == "__main__":
    asyncio.run(create_memory_collection())
