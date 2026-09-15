"""Dense sentence embeddings via Sentence-BERT.

`all-MiniLM-L6-v2` is a **bi-encoder**: it pushes each text through the
transformer independently and mean-pools the token vectors into one 384-dimension
sentence vector. Two texts are then compared by the geometry of their vectors
alone.

That independence is the whole reason this design scales. A **cross-encoder**
would read the resume and one posting *together* and score the pair, which is
more accurate but needs a full forward pass per pair — 5,000 passes per query.
A bi-encoder embeds the 5,000 postings once, offline, and a query then costs one
forward pass plus a vector search.

Unlike vanilla BERT, which was trained for masked-token prediction and whose raw
`[CLS]` vector is a poor sentence representation, Sentence-BERT is fine-tuned on
sentence pairs with a siamese objective, so cosine distance in its output space
actually tracks semantic similarity.

Vectors are L2-normalized on the way out. Once every vector has unit length,
the inner product equals the cosine similarity, which lets FAISS use its plain
`IndexFlatIP` and still rank by cosine.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL


class Embedder:
    """Turns text into unit-length dense vectors."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        # Loaded on first use: the weights are ~80MB and are downloaded once,
        # so importing this module stays cheap.
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_embedding_dimension()

    def encode_texts(
        self,
        texts,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Embed many texts at once, returning a (n, dim) float32 array."""
        texts = [text if isinstance(text, str) else "" for text in texts]
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype("float32")

    def encode_single(self, text: str) -> np.ndarray:
        """Embed one text, returning a (dim,) float32 vector."""
        return self.encode_texts([text or ""], show_progress_bar=False)[0]
