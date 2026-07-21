import pathlib
from typing import Dict, List
from src import RagDataset, StudentSearchResults, MinimalSource


def get_overlap(retrieved: MinimalSource, correct: MinimalSource) -> float:
    """Compute the character-overlap
    ratio between a retrieved and a ground-truth source."""
    end = min(retrieved.last_character_index, correct.last_character_index)
    start = max(retrieved.first_character_index, correct.first_character_index)
    overlap = max(0, end - start)
    correct_len = correct.last_character_index - correct.first_character_index

    if correct_len == 0:
        return 0.0

    return overlap / correct_len


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
             correct_sources: List[MinimalSource]) -> int:
    """Find the 1-based rank at which the first
    correct source was retrieved (11 if none in top 10)."""
    for k_index, source in enumerate(retrieved_sources):
        rank = k_index + 1
        if rank > 10:
            break
        if is_found(source, correct_sources):
            return rank
    return 11


def calcul_recall(total_k: List[int], nb_question: int) -> Dict[int, float]:
    """Aggregate per-question ranks into recall@1/3/5/10 scores."""
    result_recall: Dict[int, float] = {
        1: 0.0,
        3: 0.0,
        5: 0.0,
        10: 0.0,
    }

    for rank in total_k:
        if rank <= 1:
            result_recall[1] += 1
        if rank <= 3:
            result_recall[3] += 1
        if rank <= 5:
            result_recall[5] += 1
        if rank <= 10:
            result_recall[10] += 1

    if nb_question > 0:
        for key in result_recall.keys():
            result_recall[key] = result_recall[key] / nb_question

    return result_recall


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
    total_recall_k = []

    for q in correct:
        ground_truth[q.question_id] = q.sources

    nb_question = len(retrieved)

    for q in retrieved:
        correct_sources = ground_truth.get(q.question_id, [])
        retrieved_sources = q.retrieved_sources

        rank_found = recall_k(retrieved_sources, correct_sources)
        total_recall_k.append(rank_found)

    result_recall = calcul_recall(total_recall_k, nb_question)

    print("Evaluation Results")
    print("========================================")
    print(f"Questions evaluated: {nb_question}")
    print(f"Recall@1: {result_recall[1]:.3f}")
    print(f"Recall@3: {result_recall[3]:.3f}")
    print(f"Recall@5: {result_recall[5]:.3f}")
    print(f"Recall@10: {result_recall[10]:.3f}")
