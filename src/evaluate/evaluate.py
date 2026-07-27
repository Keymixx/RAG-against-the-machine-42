import pathlib
from typing import Dict, List
from ..models.models import RagDataset, StudentSearchResults, MinimalSource


def get_overlap(retrieved: MinimalSource, correct: MinimalSource) -> float:
    """Compute the IoU between a retrieved and a ground-truth source."""
    start = max(retrieved.first_character_index, correct.first_character_index)
    end = min(retrieved.last_character_index, correct.last_character_index)
    intersection = max(0, end - start)

    union_start = min(retrieved.first_character_index,
                      correct.first_character_index)
    union_end = max(retrieved.last_character_index,
                    correct.last_character_index)
    union = union_end - union_start

    return intersection / union if union > 0 else 0.0


def file_found(retrieved: MinimalSource, correct: MinimalSource) -> bool:
    """Check whether retrieved and correct point to the same file."""
    if retrieved.file_path == correct.file_path:
        return True
    else:
        return False


def is_found(retrieved: MinimalSource,
             correct_sources: List[MinimalSource]) -> bool:
    """Check whether retrieved sufficiently
    overlaps any ground-truth source."""
    for correct in correct_sources:
        if file_found(retrieved, correct):
            if get_overlap(retrieved, correct) >= 0.05:
                return True
    return False


def recall_k(retrieved_sources: List[MinimalSource],
             correct_sources: List[MinimalSource],
             k: int) -> float:
    """Compute recall@k for a single question."""
    if not correct_sources:
        return 0.0
    found = 0
    for correct in correct_sources:
        for source in retrieved_sources[:k]:
            if file_found(source, correct):
                if get_overlap(source, correct) >= 0.05:
                    found += 1
                    break
    return found / len(correct_sources)


def evaluate(dataset_path: str, student_answer_path: str) -> None:
    """Compute and print recall@k metrics for a student's search results.

    Raises:
        OSError: If dataset_path or student_answer_path cannot be read.
    """
    try:
        data_content = pathlib.Path(dataset_path).read_text()
        answer_content = pathlib.Path(student_answer_path).read_text()
    except OSError as exc:
        raise OSError(f"evaluate: unable to read input file: {exc}") from exc

    correct_dataset = RagDataset.model_validate_json(data_content)
    student_dataset = StudentSearchResults.model_validate_json(answer_content)

    correct = correct_dataset.rag_questions
    retrieved = student_dataset.search_results

    ground_truth = {}

    for q in correct:
        if hasattr(q, "sources"):
            ground_truth[q.question_id] = q.sources

    nb_question = len(retrieved)
    scores: Dict[int, List[float]] = {1: [], 3: [], 5: [], 10: []}

    for q in retrieved:
        correct_sources = ground_truth.get(q.question_id, [])
        retrieved_sources = q.retrieved_sources

        for k in [1, 3, 5, 10]:
            scores[k].append(recall_k(retrieved_sources, correct_sources, k))

    print("Evaluation Results")
    print("========================================")
    print(f"Questions evaluated: {nb_question}")
    for k in [1, 3, 5, 10]:
        avg = sum(scores[k]) / nb_question if nb_question > 0 else 0.0
        print(f"Recall@{k}: {avg:.3f}")