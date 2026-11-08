"""Hybrid search quality on BEIR datasets (issue #60): BM25, dense and fused.

Ranks each dataset's test queries five ways and reports nDCG@10 and
recall@100 against the human relevance judgments:
  - BM25 alone (InvertedIndex, the vectors of sparse_bm25.py)
  - dense alone (exact cosine search in a FlatIndex)
  - HybridIndex with RRF
  - HybridIndex with linear fusion at (dense, sparse) weights (1, 1), its
    default, and at the weights tune_weights picks for nDCG@10 on the test
    queries themselves: optimistic, it shows what tuning can reach.
  - The same tuning done honestly: weights tuned on the even-numbered queries,
    nDCG@10 of the odd ones, against the default on the odd ones.
Exact search on both sides keeps the comparison about fusion, not about
approximate search.

Dense vectors come from an embedding model run through fastembed (ONNX
Runtime, CPU); it is needed only here, not by qann:
    pip install fastembed
Embeddings are cached next to the dataset, so later runs take seconds.
Datasets as for sparse_bm25.py, in benchmarks/data/beir/<name>/.

    python benchmarks/hybrid_beir.py scifact fiqa
    python benchmarks/hybrid_beir.py fiqa --model BAAI/bge-small-en-v1.5
"""
import argparse
import time

import numpy as np

import qann
from sparse_bm25 import DATA, bm25, load

MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]


def embeddings(name, model_name, docs, queries):
    """(doc vectors, query vectors), from the cache or computed with fastembed."""
    tag = model_name.replace("/", "__")
    paths = [DATA / name / f"emb_{tag}_{part}.npy" for part in ("docs", "queries")]
    if all(p.exists() for p in paths):
        return [np.load(p) for p in paths]
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name)
    t = time.perf_counter()
    texts = [(d.get("title", "") + " " + d["text"]).strip() for d in docs]
    doc_vecs = np.array(list(model.passage_embed(texts, batch_size=64)), dtype=np.float32)
    query_vecs = np.array(list(model.query_embed(list(queries.values()))), dtype=np.float32)
    print(f"  embedded {len(texts):,} docs and {len(queries):,} queries in {time.perf_counter() - t:.0f} s")
    for p, v in zip(paths, (doc_vecs, query_vecs)):
        np.save(p, v)
    return doc_vecs, query_vecs


def labels(docs, queries, qrels):
    """Each query's {corpus row: grade}, as qann's ranking metrics take them."""
    row = {d["_id"]: i for i, d in enumerate(docs)}
    return [{row[d]: g for d, g in qrels[qid].items() if d in row} for qid in queries]


def scores(ids, relevant):
    """(nDCG@10, recall@100) of (nq, 100) ranked corpus rows."""
    return qann.ndcg(ids, relevant, k=10), qann.recall(ids, relevant, k=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--model", action="append", help=f"embedding model (default: {', '.join(MODELS)})")
    args = parser.parse_args()
    for name in args.datasets:
        docs, queries, qrels = load(name)
        D, Q = bm25(docs, queries)
        print(f"{name}: {len(docs):,} docs, {len(queries):,} test queries")
        sparse = qann.InvertedIndex()
        sparse.add(D)
        relevant = labels(docs, queries, qrels)
        bm25_scores = scores(sparse.batch_query(Q, 100)[0], relevant)

        for model_name in args.model or MODELS:
            print(f"  {model_name}")
            doc_vecs, query_vecs = embeddings(name, model_name, docs, queries)
            dense = qann.FlatIndex(doc_vecs.shape[1], metric=qann.Metric.Cosine)
            dense.add(doc_vecs)
            rows = {"BM25": bm25_scores, "dense": scores(dense.batch_query(query_vecs, 100)[0], relevant)}

            hybrid = qann.HybridIndex(qann.FlatIndex(doc_vecs.shape[1], metric=qann.Metric.Cosine), qann.InvertedIndex())
            hybrid.add(doc_vecs, D)   # auto ids: row i of the corpus is id i, as in the indexes above
            hybrid.method = "rrf"   # the default is "linear"
            rows["hybrid, RRF"] = scores(hybrid.batch_query(query_vecs, Q, 100)[0], relevant)
            hybrid.method = "linear"
            hybrid.weights = (1, 1)
            rows["hybrid, linear (1, 1)"] = scores(hybrid.batch_query(query_vecs, Q, 100)[0], relevant)
            hybrid.tune_weights(query_vecs, Q, relevant, 10)
            rows[f"hybrid, linear tuned {hybrid.weights} (on test)"] = scores(hybrid.batch_query(query_vecs, Q, 100)[0], relevant)

            odd = slice(1, None, 2)
            hybrid.weights = (1, 1)
            default = qann.ndcg(hybrid.batch_query(query_vecs[odd], Q[odd], 10)[0], relevant[odd])
            hybrid.tune_weights(query_vecs[::2], Q[::2], relevant[::2], 10)
            tuned = qann.ndcg(hybrid.batch_query(query_vecs[odd], Q[odd], 10)[0], relevant[odd])

            for label, (ndcg, recall) in rows.items():
                print(f"    {label:<42} nDCG@10 {ndcg:.3f}   recall@100 {recall:.3f}")
            print(f"    held out: tuned on even queries {hybrid.weights}, odd queries nDCG@10 {tuned:.3f} "
                  f"(default (1, 1): {default:.3f})")


if __name__ == "__main__":
    main()
