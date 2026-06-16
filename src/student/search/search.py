from bm25s import tokenize, BM25
from typing import List
from src.student import MinimalSource
import dspy
import json


class ResumeSignature(dspy.Signature):
    source: str = dspy.InputField(desc="source of doc")
    resume: str = dspy.OutputField(desc="short resume in one sentence")


class ResumeBot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(ResumeSignature)

    def forward(self, source: str) -> str:
        return self.prog(source=source)


def searching(query: str, k: int) -> List[MinimalSource]:

    try:
        retriever = BM25.load("data/processed/bm25s_index_vllm")
    except Exception:
        print("index file not found")

    try:
        with open("data/processed/chunks/chunks.json", "r") as f:
            chunks = json.load(f)
    except Exception:
        print("chunks file not found")

    resume_gen = ResumeBot()
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