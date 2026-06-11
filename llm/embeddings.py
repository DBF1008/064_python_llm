import collections
from .models import EmbeddingModel
from .embeddings_migrations import embeddings_migrations
from dataclasses import dataclass
import hashlib
from itertools import islice
import json
import re
from sqlite_utils import Database
from sqlite_utils.db import Table
import time
from typing import cast, Any, Dict, Iterable, List, Optional, Tuple, Union


@dataclass
class Entry:
    id: str
    score: Optional[float]
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchResult(list):
    """List subclass that carries result summary metadata alongside entries.

    Behaves exactly like a regular list of Entry objects. When a metadata
    filter was applied during the search, the ``summary`` and ``filter``
    attributes provide aggregate statistics about the filtered result set.
    """

    @property
    def summary(self) -> Optional[Dict[str, Any]]:
        return getattr(self, "_summary", None)

    @summary.setter
    def summary(self, value: Optional[Dict[str, Any]]):
        self._summary = value

    @property
    def filter(self) -> Optional[Dict[str, Any]]:
        return getattr(self, "_filter", None)

    @filter.setter
    def filter(self, value: Optional[Dict[str, Any]]):
        self._filter = value


# ---------------------------------------------------------------------------
# Metadata filter -> SQL translation
# ---------------------------------------------------------------------------

_VALID_FIELD = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_OPERATORS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in"}
_OP_MAP = {
    "$eq": "=",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}


def _filter_to_sql(filter_dict: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
    """Translate a metadata filter dict into SQLite WHERE clause fragments.

    Returns (sql_bits, args) where sql_bits is a list of SQL expressions
    to be joined with AND, and args is the corresponding list of bind
    parameters.

    Supported forms:
        {"field": value}                          -> equality
        {"field": {"$eq": value}}                 -> equality
        {"field": {"$ne": value}}                 -> inequality
        {"field": {"$gt/$gte/$lt/$lte": value}}   -> comparison
        {"field": {"$in": [v1, v2, ...]}}         -> set membership
        {"f1": v1, "f2": v2}                      -> implicit AND
    """
    if not isinstance(filter_dict, dict):
        raise ValueError("Filter must be a dict")
    if not filter_dict:
        return [], []

    sql_bits: List[str] = []
    args: List[Any] = []

    for field, condition in filter_dict.items():
        if not _VALID_FIELD.match(field):
            raise ValueError(
                f"Invalid field name: {field!r}. "
                "Field names must match [A-Za-z_][A-Za-z0-9_]* "
                "(dot-separated paths allowed)."
            )

        json_path = "$." + field
        extract = f"json_extract(metadata, '{json_path}')"

        if isinstance(condition, dict):
            for op, value in condition.items():
                if op not in _OPERATORS:
                    raise ValueError(
                        f"Unsupported operator: {op!r}. "
                        f"Supported: {', '.join(sorted(_OPERATORS))}"
                    )
                if op == "$in":
                    if not isinstance(value, (list, tuple)):
                        raise ValueError("$in requires a list value")
                    if len(value) == 0:
                        raise ValueError("$in requires a non-empty list")
                    placeholders = ", ".join("?" for _ in value)
                    sql_bits.append(f"{extract} IN ({placeholders})")
                    args.extend(value)
                else:
                    sql_op = _OP_MAP[op]
                    sql_bits.append(f"{extract} {sql_op} ?")
                    args.append(value)
        else:
            # Shorthand equality
            sql_bits.append(f"{extract} = ?")
            args.append(condition)

    return sql_bits, args


class Collection:
    class DoesNotExist(Exception):
        pass

    def __init__(
        self,
        name: str,
        db: Optional[Database] = None,
        *,
        model: Optional[EmbeddingModel] = None,
        model_id: Optional[str] = None,
        create: bool = True,
    ) -> None:
        """
        A collection of embeddings

        Returns the collection with the given name, creating it if it does not exist.

        If you set create=False a Collection.DoesNotExist exception will be raised if the
        collection does not already exist.

        Args:
            db (sqlite_utils.Database): Database to store the collection in
            name (str): Name of the collection
            model (llm.models.EmbeddingModel, optional): Embedding model to use
            model_id (str, optional): Alternatively, ID of the embedding model to use
            create (bool, optional): Whether to create the collection if it does not exist
        """
        import llm

        self.db = db or Database(memory=True)
        self.name = name
        self._model = model

        embeddings_migrations.apply(self.db)

        rows = list(self.db["collections"].rows_where("name = ?", [self.name]))
        if rows:
            row = rows[0]
            self.id = row["id"]
            self.model_id = row["model"]
        else:
            if create:
                # Collection does not exist, so model or model_id is required
                if not model and not model_id:
                    raise ValueError(
                        "Either model= or model_id= must be provided when creating a new collection"
                    )
                # Create it
                if model_id:
                    # Resolve alias
                    model = llm.get_embedding_model(model_id)
                    self._model = model
                model_id = cast(EmbeddingModel, model).model_id
                self.id = (
                    cast(Table, self.db["collections"])
                    .insert(
                        {
                            "name": self.name,
                            "model": model_id,
                        }
                    )
                    .last_pk
                )
            else:
                raise self.DoesNotExist(f"Collection '{name}' does not exist")

    def model(self) -> EmbeddingModel:
        "Return the embedding model used by this collection"
        import llm

        if self._model is None:
            self._model = llm.get_embedding_model(self.model_id)

        return cast(EmbeddingModel, self._model)

    def count(self) -> int:
        """
        Count the number of items in the collection.

        Returns:
            int: Number of items in the collection
        """
        return next(
            self.db.query(
                """
            select count(*) as c from embeddings where collection_id = (
                select id from collections where name = ?
            )
            """,
                (self.name,),
            )
        )["c"]

    def embed(
        self,
        id: str,
        value: Union[str, bytes],
        metadata: Optional[Dict[str, Any]] = None,
        store: bool = False,
    ) -> None:
        """
        Embed value and store it in the collection with a given ID.

        Args:
            id (str): ID for the value
            value (str or bytes): value to be embedded
            metadata (dict, optional): Metadata to be stored
            store (bool, optional): Whether to store the value in the content or content_blob column
        """
        from llm import encode

        content_hash = self.content_hash(value)
        if self.db["embeddings"].count_where(
            "content_hash = ? and collection_id = ?", [content_hash, self.id]
        ):
            return
        embedding = self.model().embed(value)
        cast(Table, self.db["embeddings"]).insert(
            {
                "collection_id": self.id,
                "id": id,
                "embedding": encode(embedding),
                "content": value if (store and isinstance(value, str)) else None,
                "content_blob": value if (store and isinstance(value, bytes)) else None,
                "content_hash": content_hash,
                "metadata": json.dumps(metadata) if metadata else None,
                "updated": int(time.time()),
            },
            replace=True,
        )

    def embed_multi(
        self,
        entries: Iterable[Tuple[str, Union[str, bytes]]],
        store: bool = False,
        batch_size: int = 100,
    ) -> None:
        """
        Embed multiple texts and store them in the collection with given IDs.

        Args:
            entries (iterable): Iterable of (id: str, text: str) tuples
            store (bool, optional): Whether to store the text in the content column
            batch_size (int, optional): custom maximum batch size to use
        """
        self.embed_multi_with_metadata(
            ((id, value, None) for id, value in entries),
            store=store,
            batch_size=batch_size,
        )

    def embed_multi_with_metadata(
        self,
        entries: Iterable[Tuple[str, Union[str, bytes], Optional[Dict[str, Any]]]],
        store: bool = False,
        batch_size: int = 100,
    ) -> None:
        """
        Embed multiple values along with metadata and store them in the collection with given IDs.

        Args:
            entries (iterable): Iterable of (id: str, value: str or bytes, metadata: None or dict)
            store (bool, optional): Whether to store the value in the content or content_blob column
            batch_size (int, optional): custom maximum batch size to use
        """
        import llm

        batch_size = min(batch_size, (self.model().batch_size or batch_size))
        iterator = iter(entries)
        collection_id = self.id
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            # Calculate hashes first
            items_and_hashes = [(item, self.content_hash(item[1])) for item in batch]
            # Any of those hashes already exist?
            existing_ids = [
                row["id"]
                for row in self.db.query(
                    """
                    select id from embeddings
                    where collection_id = ? and content_hash in ({})
                    """.format(",".join("?" for _ in items_and_hashes)),
                    [collection_id]
                    + [item_and_hash[1] for item_and_hash in items_and_hashes],
                )
            ]
            filtered_batch = [item for item in batch if item[0] not in existing_ids]
            embeddings = list(
                self.model().embed_multi(item[1] for item in filtered_batch)
            )
            with self.db.conn:
                cast(Table, self.db["embeddings"]).insert_all(
                    (
                        {
                            "collection_id": collection_id,
                            "id": id,
                            "embedding": llm.encode(embedding),
                            "content": (
                                value if (store and isinstance(value, str)) else None
                            ),
                            "content_blob": (
                                value if (store and isinstance(value, bytes)) else None
                            ),
                            "content_hash": self.content_hash(value),
                            "metadata": json.dumps(metadata) if metadata else None,
                            "updated": int(time.time()),
                        }
                        for (embedding, (id, value, metadata)) in zip(
                            embeddings, filtered_batch
                        )
                    ),
                    replace=True,
                )

    def similar_by_vector(
        self,
        vector: List[float],
        number: int = 10,
        skip_id: Optional[str] = None,
        prefix: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """
        Find similar items in the collection by a given vector.

        Args:
            vector (list): Vector to search by
            number (int, optional): Number of similar items to return
            skip_id (str, optional): An ID to exclude from the results
            prefix: (str, optional): Filter results to IDs with this prefix
            filter (dict, optional): Metadata filter dict, e.g.
                ``{"source": "web"}`` or ``{"year": {"$gte": 2020}}``.
                Supports operators: $eq, $ne, $gt, $gte, $lt, $lte, $in.

        Returns:
            SearchResult: List of Entry objects with optional summary metadata
        """
        import llm

        def distance_score(other_encoded):
            other_vector = llm.decode(other_encoded)
            return llm.cosine_similarity(other_vector, vector)

        self.db.register_function(distance_score, replace=True)

        where_bits = ["collection_id = ?"]
        where_args: List[Any] = [str(self.id)]

        if prefix:
            where_bits.append("id LIKE ? || '%'")
            where_args.append(prefix)

        if skip_id:
            where_bits.append("id != ?")
            where_args.append(skip_id)

        if filter:
            filter_bits, filter_args = _filter_to_sql(filter)
            where_bits.append("metadata IS NOT NULL")
            where_bits.extend(filter_bits)
            where_args.extend(filter_args)

        entries = [
            Entry(
                id=row["id"],
                score=row["score"],
                content=row["content"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in self.db.query(
                """
            select id, content, metadata, distance_score(embedding) as score
            from embeddings
            where {where}
            order by score desc limit {number}
        """.format(
                    where=" and ".join(where_bits),
                    number=number,
                ),
                where_args,
            )
        ]

        result = SearchResult(entries)

        if filter:
            result.filter = filter
            # Total count matching the filter (may exceed LIMIT)
            total_filtered = next(
                self.db.query(
                    "select count(*) as c from embeddings where {where}".format(
                        where=" and ".join(where_bits),
                    ),
                    where_args,
                )
            )["c"]
            # Facets: unique metadata value distributions in the result set
            facets: Dict[str, collections.Counter] = {}
            for entry in entries:
                if entry.metadata:
                    for key, val in entry.metadata.items():
                        facets.setdefault(key, collections.Counter())
                        facets[key][val] += 1
            scores = [e.score for e in entries if e.score is not None]
            result.summary = {
                "count": len(entries),
                "total_filtered": total_filtered,
                "score_stats": {
                    "min": min(scores) if scores else None,
                    "max": max(scores) if scores else None,
                    "avg": (sum(scores) / len(scores)) if scores else None,
                },
                "facets": {
                    k: dict(v.most_common()) for k, v in facets.items()
                },
            }

        return result

    def similar_by_id(
        self,
        id: str,
        number: int = 10,
        prefix: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """
        Find similar items in the collection by a given ID.

        Args:
            id (str): ID to search by
            number (int, optional): Number of similar items to return
            prefix: (str, optional): Filter results to IDs with this prefix
            filter (dict, optional): Metadata filter dict

        Returns:
            SearchResult: List of Entry objects with optional summary metadata
        """
        import llm

        matches = list(
            self.db["embeddings"].rows_where(
                "collection_id = ? and id = ?", (self.id, id)
            )
        )
        if not matches:
            raise self.DoesNotExist("ID not found")
        embedding = matches[0]["embedding"]
        comparison_vector = llm.decode(embedding)
        return self.similar_by_vector(
            comparison_vector, number, skip_id=id, prefix=prefix, filter=filter
        )

    def similar(
        self,
        value: Union[str, bytes],
        number: int = 10,
        prefix: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """
        Find similar items in the collection by a given value.

        Args:
            value (str or bytes): value to search by
            number (int, optional): Number of similar items to return
            prefix: (str, optional): Filter results to IDs with this prefix
            filter (dict, optional): Metadata filter dict

        Returns:
            SearchResult: List of Entry objects with optional summary metadata
        """
        comparison_vector = self.model().embed(value)
        return self.similar_by_vector(
            comparison_vector, number, prefix=prefix, filter=filter
        )

    @classmethod
    def exists(cls, db: Database, name: str) -> bool:
        """
        Does this collection exist in the database?

        Args:
            name (str): Name of the collection
        """
        rows = list(db["collections"].rows_where("name = ?", [name]))
        return bool(rows)

    def delete(self):
        """
        Delete the collection and its embeddings from the database
        """
        with self.db.conn:
            self.db.execute("delete from embeddings where collection_id = ?", [self.id])
            self.db.execute("delete from collections where id = ?", [self.id])

    @staticmethod
    def content_hash(input: Union[str, bytes]) -> bytes:
        "Hash content for deduplication. Override to change hashing behavior."
        if isinstance(input, str):
            input = input.encode("utf8")
        return hashlib.md5(input).digest()
