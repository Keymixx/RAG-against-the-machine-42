from src.chunkers import get_chunker
from transformers import AutoTokenizer
from bm25s import BM25, tokenize
import pathlib

max_token_size = 2000
model = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model)

corpus_text = []
corpus_source = []
all_path = []

path = pathlib.Path("vllm-0.10.1")
doc_path = pathlib.Path("vllm-0.10.1/docs")

all_path.extend(list(doc_path.rglob("*.md")))
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
retriever = BM25(corpus=corpus_text)
retriever.index(tokenize(corpus_text))

query = "what is vllm?"
results, scores = retriever.retrieve(tokenize(query), k=2)

# Let's see what we got!
doc, score = results[0, 0], scores[0, 0]
print(f"Rank {1} (score: {score:.2f}): {doc}")

print(f"\n\nall_files_count = {len(all_path)}")
print(f"len corpus_text = {len(corpus_text)}")