from bm25s import tokenize, BM25
from typing import List, Dict, Any
from src.student import MinimalSource
import dspy
import json


def searching(query: str, k: int,
              chunks: List[Dict[str, Any]],
              retriever: BM25) -> List[MinimalSource]:

    results, scores = retriever.retrieve(tokenize(query, stopwords="english"), k=k)
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
