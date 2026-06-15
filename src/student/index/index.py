from bm25s import BM25, tokenize
import pathlib
from src.student import get_chunker
from transformers import AutoTokenizer
from typing import List
import json
from pydantic.json import pydantic_encoder


def indexing(max_token_size: int, model="Qwen/Qwen3-0.6B"):
    max_token_size = 256
    tokenizer = AutoTokenizer.from_pretrained(model)

    corpus_text = []
    corpus_source = []
    all_path = []

    path = pathlib.Path("data/raw/vllm-0.10.1")
    doc_path = pathlib.Path("data/raw/vllm-0.10.1/docs")

    all_path.extend(list(doc_path.rglob("*.md")))
    all_path.extend(list(doc_path.rglob("vllm/*.py")))
    all_path.extend(list(path.glob("README.md")))
    # all_path.extend(list(path.rglob("*.py")))
    for path in all_path:
        chunker = get_chunker(path, tokenizer, max_token_size)
        chunks = chunker.chunk(path)
        for chunk in chunks:
            corpus_text.append(chunk.text)
            corpus_source.append(chunk)

    # print(corpus_text[0])
    # print(corpus_text[1])
    retriever = BM25()
    retriever.index(tokenize(corpus_text, stopwords="english"))
    retriever.save("data/processed/bm25s_index_vllm")

    # print(corpus_source)
    sources_chunks = json.dumps(corpus_source, default=pydantic_encoder, indent=4)
    chunks_path = pathlib.Path("data/processed/chunks/chunks.json")
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(sources_chunks)
    # with open("data/processed/chunks.json", "w+") as f:
    #     f.write(sources_chunks)
