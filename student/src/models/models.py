from pydantic import BaseModel, Field
from typing import List, Optional
import uuid


class Chunk(BaseModel):
    """A raw chunk of text extracted
    from a source file, before it is scored."""

    file_path: str
    first_character_index: int
    last_character_index: int


class MinimalSource(BaseModel):
    """A chunk of text returned by the retriever, with its rank/score."""

    file_path: str
    first_character_index: int
    last_character_index: int
    rank: Optional[int] = None
    score: Optional[str] = None


class UnansweredQuestion(BaseModel):
    """A question from the evaluation dataset, without ground-truth data."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth sources and reference answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of questions, answered or not, used to drive the CLI."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Search results (retrieved sources) for a single question."""

    question_id: str
    question_str: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results for a single
    question, augmented with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """The full set of search results
    produced by the student CLI for a dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """The full set of search results and generated answers for a dataset."""

    search_results: List[MinimalAnswer]
