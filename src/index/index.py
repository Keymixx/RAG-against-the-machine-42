import Stemmer
from bm25s import BM25, tokenize
import pathlib
from ..chunkers.chunkers import get_chunker
from ..models.models import Chunk
import json
from pydantic.json import pydantic_encoder
from tqdm import tqdm
from typing import List
import re


def expand_identifiers(text: str) -> str:
    extra = re.sub(r'[_.]', ' ', text)
    extra = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', extra)
    return text + " " + extra


def indexing(max_chunk_size: int = 2000) -> None:
    """Chunk the vLLM corpus, build a BM25 index and persist it to disk.

    Args:
        max_chunk_size: Maximum number of characters allowed per chunk.

    Raises:
        OSError: If the index or chunks cannot be saved.
    """

    corpus_text: List[str] = []
    corpus_source: List[Chunk] = []
    all_path: List[pathlib.Path] = []
    total_chunks = 0

    dir_name = ["vllm", "docs", "examples"]
    extensions = ["*.py", "*.md", "*.txt",]

    vllm_path = pathlib.Path("data/raw/vllm-0.10.1")

    for dir in dir_name:
        actual_path = vllm_path / dir
        for ext in extensions:
            all_path.extend(list(actual_path.rglob(ext)))
    for ext in extensions:
        all_path.extend(list(vllm_path.glob(ext)))

    for path in tqdm(all_path, desc="Chunking files", unit="file"):
        try:
            chunker = get_chunker(path, max_chunk_size)
            chunks, chunks_text = chunker.chunk(path)
        except (OSError, ValueError) as exc:
            print(f"indexing: skipping {path} ({exc})")
            continue
        corpus_source.extend(chunks)
        corpus_text.extend(chunks_text)
        total_chunks += len(chunks)

    retriever = BM25()
    stemmer = Stemmer.Stemmer("english")
    if not corpus_text:
        raise ValueError("indexing: no chunk produced, check the corpus path")
    retriever.index(tokenize([expand_identifiers(t) for t in corpus_text],
                             stopwords="english", stemmer=stemmer))
    try:
        retriever.save("data/processed/bm25s_index_vllm")
    except OSError as exc:
        raise OSError(f"indexing: unable to save BM25 index: {exc}") from exc

    sources_chunks = json.dumps(corpus_source,
                                default=pydantic_encoder, indent=4)

    chunks_path = pathlib.Path("data/processed/chunks/chunks.json")
    try:
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.write_text(sources_chunks)
        print(f"Ingestion complete! Index {total_chunks} chunks"
              " under data/processed/")
    except OSError as exc:
        raise OSError(f"indexing: unable to write"
                      f"{chunks_path}: {exc}") from exc
