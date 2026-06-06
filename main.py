from chonkie import CodeChunker

chunker = CodeChunker(
    language="python",      # Specify the programming language
    tokenizer="character",  # Default tokenizer (or use "gpt2", etc.)
    chunk_size=2048,        # Maximum tokens per chunk
    include_nodes=False     # Optionally include AST nodes in output
)

code = """
def hello_world():
    print("Hello, Chonkie!")

class MyClass:
    def __init__(self):
        self.value = 42
"""
chunks = chunker.chunk(code)

for chunk in chunks:
    print(f"Chunk text: {chunk.text}")
    print(f"Token count: {chunk.token_count}")