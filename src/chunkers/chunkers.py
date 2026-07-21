from abc import ABC, abstractmethod
from chonkie import RecursiveChunker, CodeChunker as CChunker
from ..models.models import Chunk
from pathlib import PosixPath
from typing import Any, List, Tuple


class BaseChunker(ABC):
    def __init__(self, max_chunk_size: int, tokenizer: Any):
        """Store the shared configuration for a chunker.

        Args:
            max_chunk_size: Maximum number of tokens allowed per chunk.
            tokenizer: Tokenizer used to measure chunk sizes.
        """
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk(self, path_file: PosixPath) -> Tuple[List[Chunk], List[str]]:
        """Split path_file into a list of Chunk and their raw text."""
        ...


class CodeChunker(BaseChunker):
    def __init__(self, max_chunk_size: int, tokenizer: Any, language: str):
        """Initialize the code chunker.

        Args:
            max_chunk_size: Maximum number of tokens allowed per chunk.
            tokenizer: Tokenizer used to measure chunk sizes.
            language: Programming language of the file (e.g. "python").
        """
        super().__init__(max_chunk_size, tokenizer)
        self.language = language
        self.tokenizer = tokenizer

    def chunk(self, path_file: PosixPath) -> Tuple[List[Chunk], List[str]]:
        """Split a source code file into chunks.

        Raises:
            OSError: If the file cannot be found or read.
            ValueError: If the file cannot be chunked.
        """
        sources: List[Chunk] = []
        sources_txt: List[str] = []

        try:
            file = path_file.read_text()
        except OSError as exc:
            raise OSError("CodeChunker: unable"
                          f"to read {path_file}: {exc}") from exc

        chunker = CChunker(
            language=self.language,
            tokenizer=self.tokenizer,
            chunk_size=self.max_chunk_size
            )

        try:
            chunks = chunker.chunk(file)
        except Exception as exc:
            raise ValueError("CodeChunker: unable"
                             f"to chunk {path_file}: {exc}") from exc
        for chunk in chunks:
            source = Chunk(
                file_path=str(path_file),
                first_character_index=chunk.start_index,
                last_character_index=chunk.end_index
            )
            sources.append(source)
            sources_txt.append(chunk.text)

        return sources, sources_txt


class MarkdownChunker(BaseChunker):
    def __init__(self, max_chunk_size: int, tokenizer: Any):
        """Initialize the markdown chunker.

        Args:
            max_chunk_size: Maximum number of tokens allowed per chunk.
            tokenizer: Tokenizer used to measure chunk sizes.
        """
        super().__init__(max_chunk_size, tokenizer)
        self.tokenizer = tokenizer

    def chunk(self, path_file: PosixPath) -> Tuple[List[Chunk], List[str]]:
        """Split a Markdown / text file into chunks.

        Raises:
            OSError: If the file cannot be found or read.
            ValueError: If the file cannot be chunked.
        """
        sources: List[Chunk] = []
        sources_txt: List[str] = []

        chunker = RecursiveChunker.from_recipe(
            name="markdown",
            tokenizer=self.tokenizer,
            chunk_size=self.max_chunk_size,
            lang="en"
        )

        try:
            file = path_file.read_text()
        except OSError as exc:
            raise OSError("MarkdownChunker: unable to read"
                          f"{path_file}: {exc}") from exc

        try:
            chunks = chunker.chunk(file)
        except Exception as exc:
            raise ValueError("MarkdownChunker: unable to chunk"
                             f"{path_file}: {exc}") from exc
        for chunk in chunks:
            source = Chunk(
                file_path=str(path_file),
                first_character_index=chunk.start_index,
                last_character_index=chunk.end_index
            )
            sources.append(source)
            sources_txt.append(chunk.text)

        return sources, sources_txt


def get_chunker(path: PosixPath,
                tokenizer: Any, max_token: int) -> BaseChunker:
    """Pick the right chunker implementation for a given file.

    Args:
        path: Path of the file that needs to be chunked.
        tokenizer: Tokenizer to hand off to the chosen chunker.
        max_token: Maximum number of tokens allowed per chunk.

    Returns:
        A CodeChunker for .py files, a MarkdownChunker otherwise.
    """
    if path.suffix == ".py":
        return CodeChunker(max_token, tokenizer, "python")
    elif path.suffix == ".md":
        return MarkdownChunker(max_token, tokenizer)
    else:
        return MarkdownChunker(max_token, tokenizer)
