from src.chunkers import get_chunker
from transformers import AutoTokenizer
import pathlib

max_token_size = 2000
model = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model)

all_sources = []
all_path = []

path = pathlib.Path("vllm-0.10.1")
all_path.extend(list(path.rglob("*.md")))
all_path.extend(list(path.rglob("*.py")))
for path in all_path:
    chunker = get_chunker(path, tokenizer, max_token_size)
    all_sources.append(chunker.chunk(path))

for sources in all_sources:
    for source in sources:
        print(source.text)
        print(f"start_index: {source.first_character_index}")
        print(f"end_index: {source.last_character_index}")
        print(f"path: {source.file_path}")
        print("\n#######################\n\n\n\n\n")