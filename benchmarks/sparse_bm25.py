"""BM25 retrieval with SparseIndex on BEIR datasets (issue #59).

Builds BM25 document vectors (Lucene's IDF, k1 = 1.2, b = 0.75, lowercase
alphanumeric tokens, Lucene's English stopwords, no stemming) and runs the
test queries one thread at a time, with MaxScore pruning off and on. Checks
that both give the same distances and reports QPS and nDCG@10 (BEIR's
Elasticsearch BM25: SciFact 0.665, FiQA 0.236).

Datasets go in benchmarks/data/beir/<name>/ (corpus.jsonl, queries.jsonl,
qrels/test.tsv), from
https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/<name>.zip

    python benchmarks/sparse_bm25.py scifact fiqa
"""
import argparse
import collections
import json
import math
import pathlib
import re
import time

import numpy as np
import scipy.sparse as sp

import qann

DATA = pathlib.Path(__file__).parent / "data" / "beir"
STOPWORDS = set("a an and are as at be but by for if in into is it no not of on or such that the their then there "
                "these they this to was will with".split())
K1, B = 1.2, 0.75


def tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def load(name):
    root = DATA / name
    docs = [json.loads(line) for line in open(root / "corpus.jsonl")]
    queries = {q["_id"]: q["text"] for q in map(json.loads, open(root / "queries.jsonl"))}
    qrels = collections.defaultdict(dict)
    for line in list(open(root / "qrels" / "test.tsv"))[1:]:
        qid, did, rel = line.split()
        qrels[qid][did] = int(rel)
    return docs, {qid: queries[qid] for qid in qrels}, qrels


def bm25(docs, queries):
    vocab = {}
    rows, cols, tfs, lengths = [], [], [], []
    for i, doc in enumerate(docs):
        toks = tokens(doc.get("title", "") + " " + doc["text"])
        lengths.append(len(toks))
        for term, tf in collections.Counter(toks).items():
            rows.append(i)
            cols.append(vocab.setdefault(term, len(vocab)))
            tfs.append(tf)
    rows, cols, tfs = np.array(rows), np.array(cols), np.array(tfs, dtype=np.float64)
    n, lengths = len(docs), np.array(lengths, dtype=np.float64)
    df = np.bincount(cols, minlength=len(vocab))
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
    norm = K1 * (1 - B + B * lengths[rows] / lengths.mean())
    weights = idf[cols] * tfs * (K1 + 1) / (tfs + norm)
    D = sp.csr_array((weights.astype(np.float32), (rows, cols)), shape=(n, len(vocab)))
    qrows, qcols = [], []
    for i, text in enumerate(queries.values()):
        for term in set(tokens(text)):
            if term in vocab:
                qrows.append(i)
                qcols.append(vocab[term])
    Q = sp.csr_array((np.ones(len(qrows), dtype=np.float32), (qrows, qcols)), shape=(len(queries), len(vocab)))
    return D, Q


def ndcg10(ids, docs, queries, qrels):
    total = 0.0
    for row, qid in zip(ids, queries):
        rel = qrels[qid]
        gains = [rel.get(docs[i]["_id"], 0) if i >= 0 else 0 for i in row[:10]]
        dcg = sum(g / math.log2(r + 2) for r, g in enumerate(gains))
        ideal = sorted(rel.values(), reverse=True)[:10]
        total += dcg / sum(g / math.log2(r + 2) for r, g in enumerate(ideal))
    return total / len(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--rounds", type=int, default=3, help="timed runs per setting; the best counts")
    args = parser.parse_args()
    qann.set_num_threads(1)
    for name in args.datasets:
        docs, queries, qrels = load(name)
        D, Q = bm25(docs, queries)
        index = qann.SparseIndex()
        index.add(D)
        print(f"{name}: {D.shape[0]:,} docs, {index.terms:,} terms, {index.postings:,} postings, {Q.shape[0]} queries")
        for k in (10, 100, 1000):
            best, results = {}, {}
            for _ in range(args.rounds):
                for pruning in (False, True):
                    index.pruning = pruning
                    t = time.perf_counter()
                    results[pruning] = index.batch_query(Q, k)
                    elapsed = time.perf_counter() - t
                    best[pruning] = min(best.get(pruning, elapsed), elapsed)
            same = np.array_equal(results[True][1], results[False][1])
            full, pruned = Q.shape[0] / best[False], Q.shape[0] / best[True]
            print(f"  k={k:<5} full {full:9,.0f} QPS   MaxScore {pruned:9,.0f} QPS  ({pruned / full:.2f}x)   "
                  f"same distances: {same}")
        print(f"  nDCG@10 {ndcg10(results[True][0], docs, queries, qrels):.3f}")


if __name__ == "__main__":
    main()
