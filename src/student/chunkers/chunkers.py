from abc import ABC, abstractmethod
from chonkie import RecursiveChunker, CodeChunker as CChunker
from src.student import MinimalSource
from pathlib import PosixPath
from typing import List


class BaseChunker(ABC):
    def __init__(self, max_chunk_size: int, tokenizer):
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk(self, path_file: str) -> List[MinimalSource]:
        ...


class CodeChunker(BaseChunker):
    def __init__(self, max_chunk_size: int, tokenizer, language: str):
        super().__init__(max_chunk_size, tokenizer)
        self.language = language
        self.tokenizer = tokenizer

    def chunk(self, path_file: PosixPath) -> List[MinimalSource]:
        sources: List[MinimalSource] = []
        
        chunker = CChunker(
            language=self.language,
            tokenizer=self.tokenizer,
            chunk_size=self.max_chunk_size
            )

        file = path_file.read_text()
        chunks = chunker.chunk(file)
        for chunk in chunks:
            source = MinimalSource(
                file_path=str(path_file),
                first_character_index=chunk.start_index,
                last_character_index=chunk.end_index,
                text=chunk.text
            )
            sources.append(source)

        return sources


class MarkdownChunker(BaseChunker):
    def __init__(self, max_chunk_size: int, tokenizer):
        super().__init__(max_chunk_size, tokenizer)
        self.tokenizer = tokenizer
    
    def chunk(self, path_file: PosixPath) -> List[MinimalSource]:
        sources: List[MinimalSource] = []

        chunker = RecursiveChunker.from_recipe(
            name="markdown",
            tokenizer=self.tokenizer,
            chunk_size=self.max_chunk_size
        )

        file = path_file.read_text()
        chunks = chunker.chunk(file)
        for chunk in chunks:
            source = MinimalSource(
                file_path=str(path_file),
                first_character_index=chunk.start_index,
                last_character_index=chunk.end_index,
                text=chunk.text
            )
            sources.append(source)

        return sources


def get_chunker(path: PosixPath, tokenizer, max_token: int) -> BaseChunker:
    if path.suffix == ".py":
        return CodeChunker(max_token, tokenizer, "python")
    elif path.suffix == ".md":
        return MarkdownChunker(max_token, tokenizer)
    else:
        return MarkdownChunker(max_token, tokenizer)
