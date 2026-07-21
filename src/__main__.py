from bm25s import BM25
import fire
import pathlib
import json
import dspy
from typing import List
from tqdm import tqdm

from .models.models import MinimalAnswer, MinimalSearchResults, RagDataset, StudentSearchResults, StudentSearchResultsAndAnswer, UnansweredQuestion
from .answer.answer_bot import AnswerBot
from .index.index import indexing
from .search.search import searching
from .evaluate.evaluate import evaluate

class RAGCLI:
    """fire-exposed CLI: index the corpus, search it, and answer questions."""

    def index(self, max_chunk_size: int) -> None:
        """Chunk the corpus and build the BM25 index."""
        indexing(max_chunk_size)

    def search(self, query: str, k: int) -> None:
        """Search the corpus and print the top k matching chunks."""
        if k <= 0:
            print("'k' is not a positive integer")
            raise SystemExit(1)
        
        try:
            retriever = BM25.load("data/processed/bm25s_index_vllm")
        except Exception as exc:
            print("index file not found")
            raise SystemExit(1) from exc

        try:
            with open("data/processed/chunks/chunks.json", "r") as f:
                chunks = json.load(f)
        except Exception as exc:
            print("chunks file not found")
            raise SystemExit(1) from exc

        sources = searching(
            query=query,
            k=k,
            chunks=chunks,
            retriever=retriever
            )

        for source in sources:
            print()
            print(f"Rank: {source.rank}\n")
            print(f"Score: {source.score}")
            print(f"File path: {source.file_path}")
            print(f"First index character: {source.first_character_index}")
            print(f"Last index character: {source.last_character_index}")
            print()

    def answer(self, query: str, k: int) -> str:
        """Search the corpus and generate a grounded answer to query."""
        if k <= 0:
            print("'k' is not a positive integer")
            raise SystemExit(1)

        try:
            retriever = BM25.load("data/processed/bm25s_index_vllm")
        except Exception as exc:
            print("index file not found")
            raise SystemExit(1) from exc

        try:
            with open("data/processed/chunks/chunks.json", "r") as f:
                chunks = json.load(f)
        except Exception as exc:
            print("chunks file not found")
            raise SystemExit(1) from exc

        answer_generator = AnswerBot()
        sources = searching(
            query=query,
            k=k,
            chunks=chunks,
            retriever=retriever
            )

        answer = answer_generator(query=query, sources=sources)
        return (answer.answer)

    def search_dataset(self, dataset_path: str,
                       k: int, save_directory: str) -> None:
        """Run search over every question of a dataset and save the results."""

        if k <= 0:
            print("'k' is not a positive integer")
            raise SystemExit(1)

        try:
            retriever = BM25.load("data/processed/bm25s_index_vllm")
        except Exception as exc:
            print("index file not found")
            raise SystemExit(1) from exc

        try:
            with open("data/processed/chunks/chunks.json", "r") as f:
                chunks = json.load(f)
        except Exception as exc:
            print("chunks file not found")
            raise SystemExit(1) from exc

        try:
            content = pathlib.Path(dataset_path).read_text()
        except OSError as exc:
            raise OSError(f"unable to read {dataset_path}: {exc}") from exc
        dataset = RagDataset.model_validate_json(content)

        rag_dataset: List[UnansweredQuestion] = dataset.rag_questions
        search_results: List[MinimalSearchResults] = []

        for q in tqdm(rag_dataset, desc="Rag dataset", unit="Question"):
            source = searching(q.question, k, chunks, retriever)
            search = MinimalSearchResults(
                question_id=q.question_id,
                question_str=q.question,
                retrieved_sources=source
            )

            search_results.append(search)

        student_result = StudentSearchResults(
            search_results=search_results,
            k=k
        )

        output_json = student_result.model_dump_json(indent=4)

        file_name = pathlib.Path(dataset_path).name
        result_path = pathlib.Path(save_directory)
        final_path = result_path / file_name
        try:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_text(output_json)
        except OSError as exc:
            raise OSError(f"unable to write {final_path}: {exc}") from exc

    def answer_dataset(self, dataset_path: str,
                       k: int, save_directory: str) -> None:
        """Run answer over every question of a dataset and save the results."""
        if k <= 0:
            print("'k' is not a positive integer")
            raise SystemExit(1)

        try:
            retriever = BM25.load("data/processed/bm25s_index_vllm")
        except Exception as exc:
            print("index file not found")
            raise SystemExit(1) from exc

        try:
            with open("data/processed/chunks/chunks.json", "r") as f:
                chunks = json.load(f)
        except Exception as exc:
            print("chunks file not found")
            raise SystemExit(1) from exc

        answer_generator = AnswerBot()

        try:
            content = pathlib.Path(dataset_path).read_text()
        except OSError as exc:
            raise OSError(f"unable to read {dataset_path}: {exc}") from exc
        dataset = RagDataset.model_validate_json(content)

        rag_dataset: List[UnansweredQuestion] = dataset.rag_questions
        search_results: List[MinimalSearchResults] = []

        for q in tqdm(rag_dataset, desc="Rag dataset", unit="Question"):
            source = searching(q.question, k, chunks, retriever)
            answer = answer_generator(query=q.question, sources=source)
            search = MinimalAnswer(
                question_id=q.question_id,
                question_str=q.question,
                retrieved_sources=source,
                answer=answer.answer
            )

            search_results.append(search)

        student_result = StudentSearchResultsAndAnswer(
            search_results=search_results,
            k=k
        )

        output_json = student_result.model_dump_json(indent=4)

        file_name = pathlib.Path(dataset_path).name
        result_path = pathlib.Path(save_directory)
        final_path = result_path / file_name
        try:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_text(output_json)
        except OSError as exc:
            raise OSError(f"unable to write {final_path}: {exc}") from exc

    def evaluate(self, dataset_path: str, student_answer_path: str) -> None:
        """Compute and print recall@k
        metrics for a saved search/answer result."""
        evaluate(dataset_path, student_answer_path)


if __name__ == "__main__":
    try:
        lm = dspy.LM(
            'ollama_chat/qwen3:0.6b',
            api_base='http://localhost:11434',
            think=False
            )

        dspy.configure(lm=lm)

        fire.Fire(RAGCLI)
    except Exception as e:
        print(e)
