import fire
from src.student import indexing, searching
import dspy



class RAGCLI:
    def index(self, max_chunk_size: int) -> None:
        indexing(max_chunk_size)

    def search(self, query: str, k: int) -> None:
        sources = searching(query=query, k=k)

        for source in sources:
            print(f"Rank: {source.rank}")
            print(f"Score: {source.score}")
            print(f"File path: {source.file_path}")
            print(f"First index character: {source.first_character_index}")
            print(f"Last index character: {source.last_character_index}")
            print(f"Resume: {source.resume}")
            print("\n\n#############################\n\n")


if __name__ == "__main__":
    lm = dspy.LM(
        'ollama_chat/qwen3:0.6b',
        api_base='http://localhost:11434',
        think=False
        )

    dspy.configure(lm=lm)

    fire.Fire(RAGCLI)
