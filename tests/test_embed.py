import json
import llm
from llm.embeddings import Entry, SearchResult, _filter_to_sql
import pytest
import sqlite_utils
from unittest.mock import ANY


def test_demo_plugin():
    model = llm.get_embedding_model("embed-demo")
    assert model.embed("hello world") == [5, 5] + [0] * 14


@pytest.mark.parametrize(
    "batch_size,expected_batches",
    (
        (None, 100),
        (10, 100),
    ),
)
def test_embed_huge_list(batch_size, expected_batches):
    model = llm.get_embedding_model("embed-demo")
    huge_list = ("hello {}".format(i) for i in range(1000))
    kwargs = {}
    if batch_size:
        kwargs["batch_size"] = batch_size
    results = model.embed_multi(huge_list, **kwargs)
    assert repr(type(results)) == "<class 'generator'>"
    first_twos = {}
    for result in results:
        key = (result[0], result[1])
        first_twos[key] = first_twos.get(key, 0) + 1
    assert first_twos == {(5, 1): 10, (5, 2): 90, (5, 3): 900}
    assert model.batch_count == expected_batches


def test_embed_store(collection):
    collection.embed("3", "hello world again", store=True)
    assert collection.db["embeddings"].count == 3
    assert (
        next(collection.db["embeddings"].rows_where("id = ?", ["3"]))["content"]
        == "hello world again"
    )


def test_embed_metadata(collection):
    collection.embed("3", "hello yet again", metadata={"foo": "bar"}, store=True)
    assert collection.db["embeddings"].count == 3
    assert json.loads(
        next(collection.db["embeddings"].rows_where("id = ?", ["3"]))["metadata"]
    ) == {"foo": "bar"}
    entry = collection.similar("hello yet again")[0]
    assert entry.id == "3"
    assert entry.metadata == {"foo": "bar"}
    assert entry.content == "hello yet again"


def test_collection(collection):
    assert collection.id == 1
    assert collection.count() == 2
    # Check that the embeddings are there
    rows = list(collection.db["embeddings"].rows)
    assert rows == [
        {
            "collection_id": 1,
            "id": "1",
            "embedding": llm.encode([5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "content": None,
            "content_blob": None,
            "content_hash": collection.content_hash("hello world"),
            "metadata": None,
            "updated": ANY,
        },
        {
            "collection_id": 1,
            "id": "2",
            "embedding": llm.encode([7, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "content": None,
            "content_blob": None,
            "content_hash": collection.content_hash("goodbye world"),
            "metadata": None,
            "updated": ANY,
        },
    ]
    assert isinstance(rows[0]["updated"], int) and rows[0]["updated"] > 0


def test_similar(collection):
    results = list(collection.similar("hello world"))
    assert results == [
        Entry(id="1", score=pytest.approx(0.9999999999999999)),
        Entry(id="2", score=pytest.approx(0.9863939238321437)),
    ]


def test_similar_prefixed(collection):
    results = list(collection.similar("hello world", prefix="2"))
    assert results == [
        Entry(id="2", score=pytest.approx(0.9863939238321437)),
    ]


def test_similar_by_id(collection):
    results = list(collection.similar_by_id("1"))
    assert results == [
        Entry(id="2", score=pytest.approx(0.9863939238321437)),
    ]


@pytest.mark.parametrize(
    "batch_size,expected_batches",
    (
        (None, 100),
        (5, 200),
    ),
)
@pytest.mark.parametrize("with_metadata", (False, True))
def test_embed_multi(with_metadata, batch_size, expected_batches):
    db = sqlite_utils.Database(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    model = collection.model()
    assert getattr(model, "batch_count", 0) == 0
    ids_and_texts = ((str(i), "hello {}".format(i)) for i in range(1000))
    kwargs = {}
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if with_metadata:
        ids_and_texts = ((id, text, {"meta": id}) for id, text in ids_and_texts)
        collection.embed_multi_with_metadata(ids_and_texts, **kwargs)
    else:
        # Exercise store=True here too
        collection.embed_multi(ids_and_texts, store=True, **kwargs)
    rows = list(db["embeddings"].rows)
    assert len(rows) == 1000
    rows_with_metadata = [row for row in rows if row["metadata"] is not None]
    rows_with_content = [row for row in rows if row["content"] is not None]
    if with_metadata:
        assert len(rows_with_metadata) == 1000
        assert len(rows_with_content) == 0
    else:
        assert len(rows_with_metadata) == 0
        assert len(rows_with_content) == 1000
    # Every row should have content_hash set
    assert all(row["content_hash"] is not None for row in rows)
    # Check batch count
    assert collection.model().batch_count == expected_batches


def test_collection_delete(collection):
    db = collection.db
    assert db["embeddings"].count == 2
    assert db["collections"].count == 1
    collection.delete()
    assert db["embeddings"].count == 0
    assert db["collections"].count == 0


def test_binary_only_and_text_only_embedding_models():
    binary_only = llm.get_embedding_model("embed-binary-only")
    text_only = llm.get_embedding_model("embed-text-only")

    assert binary_only.supports_binary
    assert not binary_only.supports_text
    assert not text_only.supports_binary
    assert text_only.supports_text

    with pytest.raises(ValueError):
        binary_only.embed("hello world")

    binary_only.embed(b"hello world")

    with pytest.raises(ValueError):
        text_only.embed(b"hello world")

    text_only.embed("hello world")

    # Try the multi versions too
    # Have to call list() on this or the generator is not evaluated
    with pytest.raises(ValueError):
        list(binary_only.embed_multi(["hello world"]))

    list(binary_only.embed_multi([b"hello world"]))

    with pytest.raises(ValueError):
        list(text_only.embed_multi([b"hello world"]))

    list(text_only.embed_multi(["hello world"]))


# ---------------------------------------------------------------------------
# Metadata filter tests
# ---------------------------------------------------------------------------


@pytest.fixture
def collection_with_metadata():
    """Collection with items having various metadata for filter testing."""
    db = sqlite_utils.Database(memory=True)
    collection = llm.Collection("test-meta", db, model_id="embed-demo")
    collection.embed(
        "1", "hello world", metadata={"source": "web", "year": 2023, "rating": 5}
    )
    collection.embed(
        "2", "goodbye world", metadata={"source": "book", "year": 2021, "rating": 3}
    )
    collection.embed(
        "3", "hello cats", metadata={"source": "web", "year": 2020, "rating": 4}
    )
    collection.embed("4", "no metadata here")  # NULL metadata
    return collection


class TestFilterToSql:
    def test_equality_shorthand(self):
        bits, args = _filter_to_sql({"source": "web"})
        assert bits == ["json_extract(metadata, '$.source') = ?"]
        assert args == ["web"]

    def test_eq_operator(self):
        bits, args = _filter_to_sql({"source": {"$eq": "web"}})
        assert bits == ["json_extract(metadata, '$.source') = ?"]
        assert args == ["web"]

    def test_ne_operator(self):
        bits, args = _filter_to_sql({"source": {"$ne": "web"}})
        assert bits == ["json_extract(metadata, '$.source') != ?"]
        assert args == ["web"]

    def test_comparison_operators(self):
        bits, args = _filter_to_sql({"year": {"$gte": 2020}})
        assert bits == ["json_extract(metadata, '$.year') >= ?"]
        assert args == [2020]

        bits, args = _filter_to_sql({"year": {"$gt": 2020}})
        assert "json_extract(metadata, '$.year') > ?" in bits

        bits, args = _filter_to_sql({"year": {"$lt": 2025}})
        assert "json_extract(metadata, '$.year') < ?" in bits

        bits, args = _filter_to_sql({"year": {"$lte": 2025}})
        assert "json_extract(metadata, '$.year') <= ?" in bits

    def test_in_operator(self):
        bits, args = _filter_to_sql({"source": {"$in": ["web", "book"]}})
        assert bits == ["json_extract(metadata, '$.source') IN (?, ?)"]
        assert args == ["web", "book"]

    def test_multiple_fields(self):
        bits, args = _filter_to_sql({"source": "web", "year": {"$gte": 2020}})
        assert len(bits) == 2
        assert "json_extract(metadata, '$.source') = ?" in bits
        assert "json_extract(metadata, '$.year') >= ?" in bits
        assert "web" in args
        assert 2020 in args

    def test_nested_path(self):
        bits, args = _filter_to_sql({"author.name": "Alice"})
        assert bits == ["json_extract(metadata, '$.author.name') = ?"]
        assert args == ["Alice"]

    def test_invalid_field_name(self):
        with pytest.raises(ValueError, match="Invalid field name"):
            _filter_to_sql({"bad field!": "value"})

    def test_invalid_operator(self):
        with pytest.raises(ValueError, match="Unsupported operator"):
            _filter_to_sql({"field": {"$invalid": "value"}})

    def test_empty_in_list(self):
        with pytest.raises(ValueError, match="non-empty list"):
            _filter_to_sql({"field": {"$in": []}})

    def test_in_requires_list(self):
        with pytest.raises(ValueError, match="requires a list"):
            _filter_to_sql({"field": {"$in": "not a list"}})

    def test_empty_filter(self):
        bits, args = _filter_to_sql({})
        assert bits == []
        assert args == []

    def test_non_dict_filter(self):
        with pytest.raises(ValueError, match="must be a dict"):
            _filter_to_sql("not a dict")


class TestSimilarWithFilter:
    def test_filter_equality(self, collection_with_metadata):
        results = collection_with_metadata.similar(
            "hello", filter={"source": "web"}
        )
        assert isinstance(results, list)
        assert isinstance(results, SearchResult)
        ids = [r.id for r in results]
        # Only items with source=web should appear (ids 1, 3)
        assert "4" not in ids  # no metadata
        assert "2" not in ids  # source=book
        assert set(ids).issubset({"1", "3"})

    def test_filter_in_operator(self, collection_with_metadata):
        results = collection_with_metadata.similar(
            "hello", filter={"source": {"$in": ["web", "book"]}}
        )
        ids = [r.id for r in results]
        assert "4" not in ids  # no metadata
        assert set(ids).issubset({"1", "2", "3"})

    def test_filter_comparison(self, collection_with_metadata):
        results = collection_with_metadata.similar(
            "hello", filter={"year": {"$gte": 2021}}
        )
        ids = [r.id for r in results]
        # Only year >= 2021: ids 1 (2023), 2 (2021)
        assert set(ids).issubset({"1", "2"})

    def test_filter_combined(self, collection_with_metadata):
        results = collection_with_metadata.similar(
            "hello", filter={"source": "web", "year": {"$gte": 2021}}
        )
        ids = [r.id for r in results]
        # source=web AND year>=2021: only id 1
        assert ids == ["1"]

    def test_filter_null_metadata_excluded(self, collection_with_metadata):
        """Items with NULL metadata should not appear when filtering."""
        results = collection_with_metadata.similar(
            "hello", filter={"source": "web"}
        )
        ids = [r.id for r in results]
        assert "4" not in ids  # id 4 has NULL metadata

    def test_filter_no_match(self, collection_with_metadata):
        results = collection_with_metadata.similar(
            "hello", filter={"source": "nonexistent"}
        )
        assert isinstance(results, SearchResult)
        assert len(results) == 0
        assert results.summary is not None
        assert results.summary["count"] == 0
        assert results.summary["total_filtered"] == 0

    def test_filter_summary_structure(self, collection_with_metadata):
        results = collection_with_metadata.similar(
            "hello", filter={"source": "web"}
        )
        assert results.summary is not None
        assert "count" in results.summary
        assert "total_filtered" in results.summary
        assert "score_stats" in results.summary
        assert "facets" in results.summary
        stats = results.summary["score_stats"]
        assert "min" in stats
        assert "max" in stats
        assert "avg" in stats

    def test_filter_summary_facets(self, collection_with_metadata):
        results = collection_with_metadata.similar(
            "hello", filter={"source": {"$in": ["web", "book"]}}
        )
        assert len(results) > 0
        assert "source" in results.summary["facets"]
        assert "year" in results.summary["facets"]

    def test_filter_is_set_on_result(self, collection_with_metadata):
        f = {"source": "web"}
        results = collection_with_metadata.similar("hello", filter=f)
        assert results.filter == f

    def test_no_filter_backward_compat(self, collection_with_metadata):
        """Without filter, results behave like a plain list."""
        results = collection_with_metadata.similar("hello")
        assert isinstance(results, list)
        assert isinstance(results, SearchResult)
        assert results.summary is None
        assert results.filter is None
        # All items including NULL metadata should appear
        ids = [r.id for r in results]
        assert "4" in ids

    def test_similar_by_id_with_filter(self, collection_with_metadata):
        results = collection_with_metadata.similar_by_id(
            "1", filter={"source": "web"}
        )
        ids = [r.id for r in results]
        assert "1" not in ids  # skip_id
        assert "2" not in ids  # source=book
        assert "4" not in ids  # no metadata
        assert set(ids).issubset({"3"})

    def test_similar_by_vector_with_filter(self, collection_with_metadata):
        model = collection_with_metadata.model()
        vector = model.embed("hello world")
        results = collection_with_metadata.similar_by_vector(
            vector, filter={"source": "web"}
        )
        ids = [r.id for r in results]
        assert "2" not in ids  # source=book
        assert "4" not in ids  # no metadata
        assert set(ids).issubset({"1", "3"})
