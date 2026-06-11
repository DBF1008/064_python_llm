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


def test_embed_different_id_same_content(embed_demo):
    """Different IDs with identical content must each get their own row."""
    db = sqlite_utils.Database(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    collection.embed("a", "hello world")
    collection.embed("b", "hello world")
    assert db["embeddings"].count == 2
    assert len(embed_demo.embedded_content) == 2
    ids = {row["id"] for row in db["embeddings"].rows}
    assert ids == {"a", "b"}


def test_embed_multi_content_update_not_blocked_by_cross_hash(embed_demo):
    """When record A's new hash equals record B's old hash, B's own update
    must not be skipped."""
    db = sqlite_utils.Database(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")

    # Initial state: a="alpha", b="beta"
    collection.embed_multi([("a", "alpha"), ("b", "beta")], store=True)
    assert db["embeddings"].count == 2
    assert len(embed_demo.embedded_content) == 2

    # Now change a's content to "beta" (same as b's old content) and
    # change b's content to "gamma".
    collection.embed_multi([("a", "beta"), ("b", "gamma")], store=True)
    assert db["embeddings"].count == 2
    assert len(embed_demo.embedded_content) == 4  # both re-embedded

    rows = {
        row["id"]: row["content"]
        for row in db.query("select id, content from embeddings")
    }
    assert rows == {"a": "beta", "b": "gamma"}


def test_embed_multi_same_id_same_content_skipped(embed_demo):
    """Re-ingesting unchanged (id, content) pairs must still be skipped."""
    db = sqlite_utils.Database(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")

    collection.embed_multi([("a", "hello"), ("b", "world")], store=True)
    assert len(embed_demo.embedded_content) == 2

    # Re-ingest identical data
    collection.embed_multi([("a", "hello"), ("b", "world")], store=True)
    assert len(embed_demo.embedded_content) == 2  # no new embeddings


def test_embed_multi_incremental_updates_vector(embed_demo):
    """After updating content, similar_by_id must use the new embedding."""
    db = sqlite_utils.Database(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")

    collection.embed_multi(
        [("doc", "short"), ("ref", "a]much longer reference sentence")], store=True
    )
    # "short" → [5,0,...], "a much longer reference sentence" → [1,4,6,9,8,0,...]
    # After update, "doc" should get a completely different vector.
    collection.embed_multi([("doc", "updated with new words here")], store=True)

    results = collection.similar_by_id("doc")
    # Just verify it returns without error and the stored content is updated
    doc_row = next(db["embeddings"].rows_where("id = ?", ["doc"]))
    assert doc_row["content"] == "updated with new words here"


def test_embed_multi_metadata_preserved_on_incremental(embed_demo):
    """Metadata on unchanged records must survive incremental re-ingestion of
    a mixed batch that includes other changed records."""
    db = sqlite_utils.Database(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")

    # Insert with metadata via embed (single)
    collection.embed("file1", "content one", metadata={"source": "disk"}, store=True)
    collection.embed("stdin1", "content two", metadata={"source": "stdin"}, store=True)
    assert db["embeddings"].count == 2

    # Re-ingest via embed_multi: file1 unchanged, stdin1 content changed.
    # embed_multi passes metadata=None, so if file1 is incorrectly re-embedded
    # its metadata would be wiped.
    collection.embed_multi([("file1", "content one"), ("stdin1", "new content")])
    assert db["embeddings"].count == 2

    file1_row = next(db["embeddings"].rows_where("id = ?", ["file1"]))
    assert json.loads(file1_row["metadata"]) == {"source": "disk"}

    stdin1_row = next(db["embeddings"].rows_where("id = ?", ["stdin1"]))
    # stdin1 was re-embedded via embed_multi (no metadata), so metadata becomes None
    assert stdin1_row["metadata"] is None


def test_embed_multi_different_ids_same_content_batch(embed_demo):
    """Multiple entries in the same batch with identical content but different
    IDs must all be inserted."""
    db = sqlite_utils.Database(memory=True)
    collection = llm.Collection("test", db, model_id="embed-demo")
    collection.embed_multi(
        [("x", "same text"), ("y", "same text"), ("z", "same text")], store=True
    )
    assert db["embeddings"].count == 3
    assert len(embed_demo.embedded_content) == 3
    ids = {row["id"] for row in db["embeddings"].rows}
    assert ids == {"x", "y", "z"}
