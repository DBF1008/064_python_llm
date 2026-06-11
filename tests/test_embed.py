import json
import llm
from llm.embeddings import Entry
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


def test_embed_same_id_different_content_updates():
    """Embedding same id with different content should update the row."""
    db = sqlite_utils.Database(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    collection.embed("1", "hello world", store=True)
    assert db["embeddings"].count == 1
    row = next(db["embeddings"].rows_where("id = ?", ["1"]))
    assert row["content"] == "hello world"
    old_hash = row["content_hash"]
    # Now update id "1" with new content
    collection.embed("1", "goodbye world", store=True)
    assert db["embeddings"].count == 1
    row = next(db["embeddings"].rows_where("id = ?", ["1"]))
    assert row["content"] == "goodbye world"
    assert row["content_hash"] != old_hash


def test_embed_different_id_same_content_both_exist(embed_demo):
    """Embedding different ids with same content should create separate rows."""
    db = sqlite_utils.Database(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    collection.embed("1", "hello world", store=True)
    assert len(embed_demo.embedded_content) == 1
    collection.embed("2", "hello world", store=True)
    # Both should exist as separate records
    assert db["embeddings"].count == 2
    assert len(embed_demo.embedded_content) == 2
    row1 = next(db["embeddings"].rows_where("id = ?", ["1"]))
    row2 = next(db["embeddings"].rows_where("id = ?", ["2"]))
    assert row1["content"] == "hello world"
    assert row2["content"] == "hello world"
    # similar_by_id should work for both
    results1 = collection.similar_by_id("1")
    assert any(r.id == "2" for r in results1)
    results2 = collection.similar_by_id("2")
    assert any(r.id == "1" for r in results2)


def test_embed_same_id_same_content_skips(embed_demo):
    """Re-embedding same id with same content should not call the model."""
    db = sqlite_utils.Database(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    collection.embed("1", "hello world")
    assert len(embed_demo.embedded_content) == 1
    # Embed again with same id and same content
    collection.embed("1", "hello world")
    assert db["embeddings"].count == 1
    # Model should NOT have been called again
    assert len(embed_demo.embedded_content) == 1


def test_embed_multi_same_id_different_content_updates():
    """Batch embedding existing id with new content should update the row."""
    db = sqlite_utils.Database(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    collection.embed("1", "hello world", store=True)
    row = next(db["embeddings"].rows_where("id = ?", ["1"]))
    old_hash = row["content_hash"]
    # Now batch embed with same id but different content
    collection.embed_multi([("1", "goodbye world")], store=True)
    assert db["embeddings"].count == 1
    row = next(db["embeddings"].rows_where("id = ?", ["1"]))
    assert row["content"] == "goodbye world"
    assert row["content_hash"] != old_hash


def test_embed_multi_different_id_same_content_both_exist(embed_demo):
    """Batch embedding new ids with existing content should create separate rows."""
    db = sqlite_utils.Database(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    collection.embed("1", "hello world", store=True)
    assert len(embed_demo.embedded_content) == 1
    # Batch embed same content under different ids
    collection.embed_multi(
        [("2", "hello world"), ("3", "hello world")], store=True
    )
    assert db["embeddings"].count == 3
    assert len(embed_demo.embedded_content) == 3
    for rid in ("1", "2", "3"):
        row = next(db["embeddings"].rows_where("id = ?", [rid]))
        assert row["content"] == "hello world"


def test_embed_multi_re_run_idempotent(embed_demo):
    """Running same batch twice should skip all on second run."""
    db = sqlite_utils.Database(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    batch = [("1", "hello world"), ("2", "goodbye world"), ("3", "foo bar")]
    collection.embed_multi(batch, store=True)
    assert db["embeddings"].count == 3
    assert len(embed_demo.embedded_content) == 3
    # Run same batch again
    collection.embed_multi(batch, store=True)
    assert db["embeddings"].count == 3
    # Model should NOT have been called again
    assert len(embed_demo.embedded_content) == 3
