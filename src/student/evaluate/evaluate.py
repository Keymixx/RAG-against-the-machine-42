from src.student import AnsweredQuestion, RagDataset, MinimalSource
import pathlib


def evaluate(dataset_path: str, answer_path: str):
    content = pathlib.Path(dataset_path).read_text()
    dataset = RagDataset.model_validate_json(content)

    rag_dataset = dataset.rag_questions

    # for q in rag_dataset:

def get_overlap(retrieved: MinimalSource, correct: MinimalSource):
    end = min(retrieved.last_character_index, correct.last_character_index)
    start = max(retrieved.first_character_index, correct.first_character_index)
    overlap = max(0, end - start)
    correct_len = correct.last_character_index - correct.first_character_index
    print(start)
    print(end)
    return overlap / correct_len
