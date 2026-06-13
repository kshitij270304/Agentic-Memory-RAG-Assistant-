import unittest
from unittest.mock import patch

from mem.vectordb import (
    EmbeddedMemory,
    _get_collection,
    _deserialize_categories,
    _metadata_to_memory,
    _serialize_categories,
    insert_memories,
    search_memories,
)


class FakeCollection:
    def __init__(self):
        self.upsert_args = None
        self.query_args = None

    def upsert(self, **kwargs):
        self.upsert_args = kwargs

    def query(self, **kwargs):
        self.query_args = kwargs
        return {
            "ids": [["football", "music"]],
            "documents": [[
                "My favorite sport is football.",
                "My favorite music is jazz.",
            ]],
            "metadatas": [[
                {
                    "user_id": 1,
                    "categories": '["sport","preference"]',
                    "date": "2026-06-13 10:00",
                },
                {
                    "user_id": 1,
                    "categories": '["music","preference"]',
                    "date": "2026-06-13 10:01",
                },
            ]],
            "distances": [[0.1, 0.2]],
        }


class FakeClient:
    def __init__(self):
        self.collection_args = None
        self.collection = FakeCollection()

    def get_or_create_collection(self, **kwargs):
        self.collection_args = kwargs
        return self.collection


class VectorDbHelpersTests(unittest.TestCase):
    def tearDown(self):
        import mem.vectordb as vectordb

        vectordb._collection = None

    def test_collection_uses_cosine_and_external_embeddings(self):
        client = FakeClient()

        with patch(
            "mem.vectordb._get_client",
            return_value=client,
        ):
            collection = _get_collection()

        self.assertIs(collection, client.collection)
        self.assertEqual(
            client.collection_args["configuration"],
            {
                "hnsw": {
                    "space": "cosine",
                }
            },
        )
        self.assertIsNone(
            client.collection_args["embedding_function"]
        )

    def test_categories_round_trip_as_scalar_metadata(self):
        serialized = _serialize_categories(
            ["Sport", "Preference"]
        )

        self.assertEqual(
            serialized,
            '["sport","preference"]',
        )
        self.assertEqual(
            _deserialize_categories(serialized),
            ["sport", "preference"],
        )

    def test_cosine_distance_becomes_relevance_score(self):
        memory = _metadata_to_memory(
            point_id="memory-1",
            document="The user likes football.",
            metadata={
                "user_id": 1,
                "categories": '["sport"]',
                "date": "2026-06-13",
            },
            distance=0.15,
        )

        self.assertAlmostEqual(memory.score, 0.85)


class VectorDbOperationsTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_insert_sends_embeddings_and_scalar_metadata(self):
        collection = FakeCollection()

        with patch(
            "mem.vectordb._get_collection",
            return_value=collection,
        ):
            await insert_memories([
                EmbeddedMemory(
                    user_id=1,
                    memory_text="My favorite sport is football.",
                    categories=["sport", "preference"],
                    date="2026-06-13 10:00",
                    embedding=[0.1, 0.2],
                )
            ])

        metadata = collection.upsert_args[
            "metadatas"
        ][0]
        self.assertEqual(
            metadata["categories"],
            '["sport","preference"]',
        )
        self.assertEqual(
            collection.upsert_args["embeddings"],
            [[0.1, 0.2]],
        )

    async def test_search_filters_categories_after_query(self):
        collection = FakeCollection()

        with patch(
            "mem.vectordb._get_collection",
            return_value=collection,
        ):
            results = await search_memories(
                search_vector=[0.1, 0.2],
                user_id=1,
                categories=["sport"],
            )

        self.assertEqual(
            [memory.point_id for memory in results],
            ["football"],
        )
        self.assertEqual(
            collection.query_args["where"],
            {"user_id": 1},
        )


if __name__ == "__main__":
    unittest.main()
