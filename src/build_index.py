"""One-time script: embed the job corpus and build the FAISS index.

    python -m src.build_index

`IndexFlatIP` is an exhaustive index — it scores the query against every stored
vector. That is the right choice here: 5,000 vectors of 384 dimensions is about
7MB, a query takes a couple of milliseconds, and the results are exact. An
approximate index (IVF, HNSW) trades some recall for speed and only starts to
pay off at hundreds of thousands of vectors.

"IP" is inner product. Because `Embedder` returns unit-length vectors, the inner
product is exactly the cosine similarity.
"""

import pickle
import time
from datetime import datetime, timezone

import faiss
import numpy as np
import pandas as pd

from src.config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    JOBS_META_PATH,
    JOBS_PARQUET,
    MODELS_DIR,
)
from src.embedder import Embedder


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Wrap normalized vectors in an exact inner-product index."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def main() -> None:
    jobs = pd.read_parquet(JOBS_PARQUET, columns=["job_id", "job_text"])
    texts = jobs["job_text"].fillna("").tolist()
    print(f"Embedding {len(texts):,} postings with {EMBEDDING_MODEL}…")

    embedder = Embedder()
    started = time.perf_counter()
    vectors = embedder.encode_texts(texts)
    embed_seconds = time.perf_counter() - started

    started = time.perf_counter()
    index = build_index(vectors)
    index_seconds = time.perf_counter() - started

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    # The index stores vectors only, so the row order that maps a FAISS result
    # back to a posting is recorded separately and checked at load time.
    metadata = {
        "job_ids": jobs["job_id"].tolist(),
        "model_name": EMBEDDING_MODEL,
        "dimension": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Recorded here so the benchmark can report offline build cost next to
        # query latency without re-running the build.
        "embed_seconds": round(embed_seconds, 1),
        "index_seconds": round(index_seconds, 4),
        "index_bytes": FAISS_INDEX_PATH.stat().st_size,
    }
    with open(JOBS_META_PATH, "wb") as handle:
        pickle.dump(metadata, handle)

    size_mb = FAISS_INDEX_PATH.stat().st_size / 1e6
    print(f"Vectors:      {vectors.shape[0]:,} x {vectors.shape[1]}")
    print(f"Embedding:    {embed_seconds:.1f}s ({len(texts) / embed_seconds:.0f} texts/s)")
    print(f"Index build:  {index_seconds * 1000:.0f}ms")
    print(f"Index size:   {size_mb:.1f} MB")
    print(f"Saved index:  {FAISS_INDEX_PATH}")
    print(f"Saved meta:   {JOBS_META_PATH}")


if __name__ == "__main__":
    main()
