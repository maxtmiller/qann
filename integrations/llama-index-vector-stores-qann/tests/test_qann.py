import zlib

import numpy as np
import pytest
import qann
from llama_index.core import Document, MockEmbedding, Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from llama_index.vector_stores.qann import QANNVectorStore

DIM = 16
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()


def vec(text):
    v = np.random.default_rng(zlib.crc32(text.encode())).standard_normal(DIM)
    return (v / np.linalg.norm(v)).tolist()


def nodes(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        text = " ".join(rng.choice(WORDS, rng.integers(3, 10))) + f" unique{i}"
        meta = {"year": int(2000 + i % 10), "source": ["a", "b", "c"][i % 3], "score": float(i % 7) / 2}
        if i % 4 == 0:
            del meta["source"]   # some nodes lack a key
        node = TextNode(text=text, id_=f"n{i}", metadata=meta, embedding=vec(text))
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=f"doc{i // 5}")
        out.append(node)
    return out


def query(store, embedding, k, **kw):
    return store.query(VectorStoreQuery(query_embedding=embedding, similarity_top_k=k, **kw))


F = MetadataFilter
OPS = [
    F(key="year", value=2003, operator=FilterOperator.EQ),
    F(key="year", value=2003, operator=FilterOperator.NE),
    F(key="year", value=2005, operator=FilterOperator.GT),
    F(key="year", value=2005, operator=FilterOperator.GTE),
    F(key="score", value=1.5, operator=FilterOperator.LT),
    F(key="score", value=1.5, operator=FilterOperator.LTE),
    F(key="source", value="b", operator=FilterOperator.EQ),
    F(key="source", value="b", operator=FilterOperator.NE),
    F(key="source", value=["a", "c"], operator=FilterOperator.IN),
    F(key="source", value=["a", "c"], operator=FilterOperator.NIN),
    F(key="source", value=None, operator=FilterOperator.IS_EMPTY),
    F(key="missing", value="x", operator=FilterOperator.EQ),
    F(key="missing", value="x", operator=FilterOperator.NE),
    F(key="year", value=[2001, 2002], operator=FilterOperator.IN),
]


def random_filters(rng):
    pick = [OPS[i] for i in rng.choice(len(OPS), rng.integers(1, 4), replace=False)]
    return MetadataFilters(filters=pick, condition=[FilterCondition.AND, FilterCondition.OR, FilterCondition.NOT][rng.integers(3)])


def test_filters_match_llamaindex_reference():
    data = nodes(120)
    ours = QANNVectorStore(index="flat")
    ours.add(data)
    ref = SimpleVectorStore()
    ref.add(data)
    rng = np.random.default_rng(3)
    q = vec("probe")
    for _ in range(150):
        filters = random_filters(rng)
        got = {n.node_id for n in query(ours, q, 200, filters=filters).nodes}
        want = set(ref.query(VectorStoreQuery(query_embedding=q, similarity_top_k=200, filters=filters)).ids)
        assert got == want, filters
        # get_nodes / delete_nodes filter in Python with the same semantics
        assert {n.node_id for n in ours.get_nodes(filters=filters)} == want


def test_nested_filters():
    store = QANNVectorStore(index="flat")
    data = nodes(60)
    store.add(data)
    filters = MetadataFilters(condition=FilterCondition.OR, filters=[
        MetadataFilters(filters=[F(key="year", value=2001), F(key="source", value="b")]),
        F(key="score", value=3.0, operator=FilterOperator.GTE),
    ])
    want = {n.node_id for n in data if (n.metadata["year"] == 2001 and n.metadata.get("source") == "b")
            or n.metadata["score"] >= 3.0}
    assert {n.node_id for n in query(store, vec("x"), 100, filters=filters).nodes} == want
    assert {n.node_id for n in store.get_nodes(filters=filters)} == want


def test_unsupported_operators_raise():
    store = QANNVectorStore(index="flat")
    store.add(nodes(5))
    for op in (FilterOperator.TEXT_MATCH, FilterOperator.CONTAINS, FilterOperator.ANY, FilterOperator.ALL):
        with pytest.raises(NotImplementedError):
            query(store, vec("x"), 3, filters=MetadataFilters(filters=[F(key="source", value=["a"] if op in (FilterOperator.ANY, FilterOperator.ALL) else "a", operator=op)]))


def test_nodes_round_trip_and_delete_by_document():
    data = nodes(20)
    store = QANNVectorStore()
    assert store.add(data) == [n.node_id for n in data]
    got = store.get_nodes(node_ids=["n7"])[0]
    assert got.get_content() == data[7].get_content() and got.metadata == data[7].metadata
    assert got.ref_doc_id == "doc1"
    store.delete("doc1")   # nodes n5-n9
    left = {n.node_id for n in store.get_nodes()}
    assert left == {f"n{i}" for i in range(20)} - {f"n{i}" for i in range(5, 10)}
    store.delete_nodes(node_ids=["n0", "n1"])
    store.delete_nodes(filters=MetadataFilters(filters=[F(key="year", value=2003)]))
    assert {n.node_id for n in store.get_nodes()} == left - {"n0", "n1", "n3", "n13"}
    store.add([data[0]])   # re-add
    store.add([TextNode(text="changed", id_="n0", embedding=vec("changed"))])   # replace
    assert store.get_nodes(node_ids=["n0"])[0].get_content() == "changed"
    store.clear()
    assert store.get_nodes() == []


def test_query_restrictions_and_similarities():
    data = nodes(50)
    store = QANNVectorStore(index="flat")
    store.add(data)
    res = query(store, data[10].embedding, 3)
    assert res.ids[0] == "n10" and res.similarities[0] == pytest.approx(1.0, abs=1e-5)
    assert res.similarities == sorted(res.similarities, reverse=True)
    assert set(query(store, data[10].embedding, 10, node_ids=["n1", "n2", "n3"]).ids) == {"n1", "n2", "n3"}
    assert set(query(store, data[10].embedding, 10, doc_ids=["doc2"]).ids) == {f"n{i}" for i in range(10, 15)}
    assert query(store, data[10].embedding, 10, node_ids=["n1"], doc_ids=["doc2"]).ids == []
    mmr = store.query(VectorStoreQuery(query_embedding=data[10].embedding, similarity_top_k=4,
                                       mode=VectorStoreQueryMode.MMR, mmr_threshold=0.3))
    assert len(mmr.ids) == 4 and len(set(mmr.ids)) == 4


@pytest.mark.parametrize("preset", ["hnsw", "flat", "ivf-pq"])
def test_presets_and_search_settings(preset):
    data = nodes(1500)
    store = QANNVectorStore(index=preset, search_params={"nprobe": 32} if preset == "ivf-pq" else {})
    store.add(data)
    hits = sum(query(store, n.embedding, 1).ids[0] == n.node_id for n in data[:100])
    assert hits >= 95
    if preset == "hnsw":
        store.query(VectorStoreQuery(query_embedding=data[0].embedding, similarity_top_k=5), ef_search=128)
        assert store.client.ef_search != 128
        with pytest.raises(ValueError, match="no search setting"):
            store.query(VectorStoreQuery(query_embedding=data[0].embedding), nprobe=4)


def test_hybrid_modes():
    data = nodes(300)
    store = QANNVectorStore(sparse="bm25")
    store.add(data)
    q = VectorStoreQuery(query_embedding=vec("anything"), query_str="unique123", similarity_top_k=1,
                         mode=VectorStoreQueryMode.SPARSE)
    assert store.query(q).ids == ["n123"]
    # LlamaIndex's retriever sends sparse-mode queries without an embedding
    assert store.query(VectorStoreQuery(query_str="unique123", similarity_top_k=1,
                                        mode=VectorStoreQueryMode.SPARSE)).ids == ["n123"]
    with pytest.raises(ValueError, match="needs query_embedding"):
        store.query(VectorStoreQuery(query_str="unique123", similarity_top_k=1))
    q.mode = VectorStoreQueryMode.HYBRID
    assert store.query(q).ids == ["n123"]
    q.alpha = 1.0   # dense only: a random query vector won't find it
    assert store.query(q).ids != ["n123"]
    assert store.client.weights == (1.0, 1.0)   # restored
    q = VectorStoreQuery(query_embedding=data[5].embedding, query_str="unique5", similarity_top_k=1)
    assert store.query(q).ids == ["n5"]   # default mode: dense side
    table = store.tune_hybrid_weights([vec(f"q{i}") for i in range(0, 300, 10)],
                                      [f"unique{i}" for i in range(0, 300, 10)],
                                      [{f"n{i}"} for i in range(0, 300, 10)])
    assert max(table.values()) == pytest.approx(1.0) and store.client.weights[1] > 0
    with pytest.raises(ValueError, match="needs hybrid search"):
        plain = QANNVectorStore()
        plain.add(data[:3])
        plain.query(VectorStoreQuery(query_embedding=vec("x"), query_str="x", mode=VectorStoreQueryMode.HYBRID))


@pytest.mark.parametrize("mmap", [False, True])
def test_persist_and_load(tmp_path, mmap):
    data = nodes(1200)
    store = QANNVectorStore(index="ivf-pq", sparse="bm25")
    store.add(data)
    store.delete("doc3")
    path = str(tmp_path / "store" / "vectors.qann")
    store.persist(path)
    loaded = QANNVectorStore.from_persist_path(path, mmap=mmap)
    assert loaded.client.dense.mapped == mmap
    for q in [data[0].embedding, data[500].embedding]:
        a = loaded.query(VectorStoreQuery(query_embedding=q, query_str="alpha kilo", similarity_top_k=5,
                                          mode=VectorStoreQueryMode.HYBRID))
        b = store.query(VectorStoreQuery(query_embedding=q, query_str="alpha kilo", similarity_top_k=5,
                                         mode=VectorStoreQueryMode.HYBRID))
        assert a.ids == b.ids
    assert loaded.get_nodes(node_ids=["n16"]) == []   # doc3 stayed deleted
    loaded.delete("doc4")
    assert loaded.get_nodes(node_ids=["n20"]) == []


def test_vector_store_index_end_to_end(tmp_path):
    Settings.embed_model = MockEmbedding(embed_dim=DIM)
    docs = [Document(text=f"document {i} about {WORDS[i % len(WORDS)]}", id_=f"d{i}", metadata={"i": i}) for i in range(30)]
    store = QANNVectorStore(sparse="bm25")
    ctx = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex.from_documents(docs, storage_context=ctx)
    retriever = index.as_retriever(similarity_top_k=3, vector_store_query_mode="hybrid", alpha=0.0,
                                   filters=MetadataFilters(filters=[F(key="i", value=10, operator=FilterOperator.GTE)]))
    got = retriever.retrieve("juliet")
    assert got and all(r.node.metadata["i"] >= 10 for r in got) and "juliet" in got[0].node.get_content()
    index.delete_ref_doc("d25")
    assert all(n.ref_doc_id != "d25" for n in store.get_nodes())
    ctx.persist(persist_dir=str(tmp_path))
    reopened = QANNVectorStore.from_persist_path(str(tmp_path / "default__vector_store.json"))
    again = VectorStoreIndex.from_vector_store(reopened).as_retriever(
        similarity_top_k=3, vector_store_query_mode="hybrid", alpha=0.0).retrieve("juliet")
    assert [r.node.node_id for r in again] == [r.node.node_id for r in index.as_retriever(
        similarity_top_k=3, vector_store_query_mode="hybrid", alpha=0.0).retrieve("juliet")]


async def test_async_methods():
    data = nodes(10)
    store = QANNVectorStore(index="flat")
    assert await store.async_add(data) == [n.node_id for n in data]
    res = await store.aquery(VectorStoreQuery(query_embedding=data[2].embedding, similarity_top_k=1))
    assert res.ids == ["n2"]
    await store.adelete("doc0")
    assert {n.node_id for n in await store.aget_nodes()} == {f"n{i}" for i in range(5, 10)}


def test_own_index_and_errors():
    store = QANNVectorStore(index=qann.HNSWIndex(DIM, M=8, metric=qann.Metric.L2))
    store.add(nodes(5))
    res = query(store, nodes(5)[1].embedding, 1)
    assert res.ids == ["n1"] and res.similarities[0] == pytest.approx(0, abs=1e-5)   # -distance for L2
    foreign = qann.FlatIndex(DIM)
    foreign.add(np.ones((1, DIM), dtype=np.float32))
    with pytest.raises(ValueError, match="outside a QANNVectorStore"):
        QANNVectorStore(index=foreign)
    with pytest.raises(ValueError, match="at least 1000"):
        QANNVectorStore(index="ivf-pq").add(nodes(10))
