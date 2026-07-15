import dspy
from student.src import MinimalSource, get_text
from typing import List


class AnswerSignature(dspy.Signature):
    """Answer a question about the vLLM codebase using only the
    provided sources."""

    query: str = dspy.InputField(
        desc="The user's question about the vLLM codebase or "
             "documentation"
    )
    sources: List[str] = dspy.InputField(
        desc="Retrieved code snippets and documentation excerpts "
             "relevant to the query"
    )
    answer: str = dspy.OutputField(
        desc=(
            "A self-contained answer that fully makes sense without "
            "seeing the original question. Must be grounded strictly "
            "in the provided sources with no outside knowledge or "
            "hallucination. Cite the relevant file or function name "
            "when applicable. Write in plain text only, do not use "
            "LaTeX or markdown math notation. Directly and concisely "
            "answer what was asked."
        )
    )


class AnswerBot(dspy.Module):
    """Generates an answer to a query,
    grounded in a list of retrieved sources."""

    def __init__(self) -> None:
        """Set up the underlying DSPy predictor."""
        super().__init__()
        self.prog = dspy.Predict(AnswerSignature)

    def forward(self, query: str,
                sources: List[MinimalSource]) -> dspy.Prediction:
        """Generate a grounded answer for query from sources.

        Raises:
            OSError: If a source file cannot be found or read.
        """
        sources_txt: List[str] = []
        for source in sources:
            try:
                sources_txt.append(get_text(
                    file_path=source.file_path,
                    first_index=source.first_character_index,
                    last_index=source.last_character_index
                ))
            except OSError as exc:
                raise OSError("AnswerBot: unable to load source"
                              f"{source.file_path}: {exc}") from exc
        return self.prog(query=query, sources=sources_txt)
