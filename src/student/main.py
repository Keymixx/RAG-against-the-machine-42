from bm25s import BM25
import fire
import pathlib
from pydantic.json import pydantic_encoder
from src.student import indexing, searching, MinimalSource, StudentSearchResults
from src.student import AnswerBot, RagDataset, UnansweredQuestion, MinimalSearchResults
import json
import dspy
from typing import List


class RAGCLI:
    def index(self, max_chunk_size: int) -> None:
        indexing(max_chunk_size)

    def search(self, query: str, k: int) -> None:
        try:
            retriever = BM25.load("data/processed/bm25s_index_vllm")
        except Exception:
            print("index file not found")

        try:
            with open("data/processed/chunks/chunks.json", "r") as f:
                chunks = json.load(f)
        except Exception:
            print("chunks file not found")

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

    def answer(self, query: str, k: int) -> None:
        answer_generator = AnswerBot()
        sources = searching(query=query, k=k)

        answer = answer_generator(query=query, sources=sources)
        print(f"answer: {answer.answer}")

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> None:
        try:
            retriever = BM25.load("data/processed/bm25s_index_vllm")
        except Exception:
            print("index file not found")

        try:
            with open("data/processed/chunks/chunks.json", "r") as f:
                chunks = json.load(f)
        except Exception:
            print("chunks file not found")

        content = pathlib.Path(dataset_path).read_text()
        dataset = RagDataset.model_validate_json(content)

        rag_dataset: List[UnansweredQuestion] = dataset.rag_questions
        search_results: List[MinimalSearchResults] = []

        for q in rag_dataset:
            source = searching(q.question_str, k, chunks, retriever)
            search = MinimalSearchResults(
                question_id=q.question_id,
                question_str=q.question_str,
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
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text(output_json)


if __name__ == "__main__":
    lm = dspy.LM(
        'ollama_chat/qwen3:0.6b',
        api_base='http://localhost:11434',
        think=False
        )

    dspy.configure(lm=lm)

    fire.Fire(RAGCLI)
