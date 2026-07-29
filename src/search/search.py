from bm25s import tokenize, BM25
import Stemmer
from typing import List, Dict, Any
from ..models.models import MinimalSource
import re

def expand_identifiers(text: str) -> str:
    extra = re.sub(r'[_.]', ' ', text)
    extra = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', extra)
    return text + " " + extra

def searching(query: str, k: int,
              chunks: List[Dict[str, Any]],
              retriever: BM25) -> List[MinimalSource]:
    """Retrieve the k most relevant chunks for a query using BM25.

    Args:
        query: The natural language query to search for.
        k: Number of results to retrieve.
        chunks: The full corpus of chunk metadata.
        retriever: A loaded/fitted bm25s.BM25 index.

    Returns:
        A list of MinimalSource, ranked from most to least relevant.

    Raises:
        ValueError: If the retrieval step fails.
    """
    try:
        stemmer = Stemmer.Stemmer("english")
        results, scores = retriever.retrieve(
            tokenize(expand_identifiers(query), stopwords="english", stemmer=stemmer), k=k)
    except Exception as exc:
        raise ValueError(f"searching: BM25 retrieval"
                         f"failed for query {query!r}: {exc}") from exc

    sources: List[MinimalSource] = []
    for i in range(results.shape[1]):
        doc_i = results[0, i]
        score = scores[0, i]
        sources.append(MinimalSource(
            **chunks[doc_i],
            rank=(1+i),
            score=f"{score:.2f}",
            ))

    return sources
