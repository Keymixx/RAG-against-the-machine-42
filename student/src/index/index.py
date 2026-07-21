from bm25s import BM25, tokenize
import pathlib
from src import get_chunker, Chunk
from transformers import AutoTokenizer
import json
from pydantic.json import pydantic_encoder
from tqdm import tqdm
from typing import List


def indexing(max_token_size: int, model: str = "Qwen/Qwen3-0.6B") -> None:
    """Chunk the vLLM corpus, build a
    BM25 index and persist everything to disk.

    Args:
        max_token_size: Maximum number of tokens allowed per chunk.
        model: Hugging Face model id used to load the tokenizer.

    Raises:
        OSError: If the tokenizer, index or chunks cannot be loaded/saved.
    """
    max_token_size = 256
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
    except Exception as exc:
        raise OSError(f"indexing: unable to"
                      f"load tokenizer {model}: {exc}") from exc

    corpus_text: List[str] = []
    corpus_source: List[Chunk] = []
    all_path: List[pathlib.Path] = []

    dir_name = ["vllm", "docs"]
    extensions = ["*.py", "*.md", "*.txt",]

    vllm_path = pathlib.Path("data/raw/vllm-0.10.1")
    all_path = []

    for dir in dir_name:
        actual_path = vllm_path / dir
        for ext in extensions:
            all_path.extend(list(actual_path.rglob(ext)))
    for ext in extensions:
        all_path.extend(list(vllm_path.glob(ext)))

    for path in tqdm(all_path, desc="Chunking files", unit="file"):
        try:
            chunker = get_chunker(path, tokenizer, max_token_size)
            chunks, chunks_text = chunker.chunk(path)
        except (OSError, ValueError) as exc:
            print(f"indexing: skipping {path} ({exc})")
            continue
        corpus_source.extend(chunks)
        corpus_text.extend(chunks_text)

    retriever = BM25()
    retriever.index(tokenize(corpus_text, stopwords="english"))
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
    except OSError as exc:
        raise OSError(f"indexing: unable to write"
                      f"{chunks_path}: {exc}") from exc
