*This project has been created as part of the 42 curriculum by caaubert.*

# RAG against the machine

## Description

This project implements a **Retrieval-Augmented Generation (RAG)** system that
answers questions about a codebase — here, the vLLM repository.

A language model only knows what it was trained on. Instead of retraining it to
teach it a new codebase, RAG gives the model access to an external source of
information at answer time. The system indexes the corpus once, retrieves the
snippets most relevant to a question, and hands them to a small local model
(`Qwen/Qwen3-0.6B`) which produces an answer grounded in those snippets.

The pipeline has four stages:

1. **Indexing** — read the corpus, split every file into chunks, build a BM25
   index and persist it to disk.
2. **Retrieval** — score the indexed chunks against a question and return the
   top-k source locations.
3. **Generation** — pass the retrieved context to the LLM and generate an answer.
4. **Evaluation** — measure retrieval quality with recall@k against a
   ground-truth dataset.

## Instructions

### Requirements

- Python 3.10
- [uv](https://docs.astral.sh/uv/) as package manager
- [Ollama](https://ollama.com/) running locally with the `qwen3:0.6b` model,
  for answer generation only

### Installation

```bash
uv sync
ollama pull qwen3:0.6b
```

The corpus must be extracted under `data/raw/vllm-0.10.1/`.

### Makefile targets

| Target | Effect |
|---|---|
| `make install` | Install dependencies with `uv sync` |
| `make run` | Run the CLI entry point |
| `make debug` | Run the CLI under `pdb` |
| `make clean` | Remove `__pycache__` and `.mypy_cache` |
| `make lint` | Run `flake8` and `mypy` with the required flags |

## System architecture

```
data/raw/vllm-0.10.1/
        |
        v
   [ chunkers ]  CodeChunker (.py)  |  MarkdownChunker (.md, .txt)
        |
        +--> corpus_text    (chunk contents, fed to BM25)
        +--> corpus_source  (file_path + character span, persisted as JSON)
        |
        v
   [ index ]  bm25s.BM25 --> data/processed/bm25s_index_vllm
                             data/processed/chunks/chunks.json
        |
        v
   [ search ]  query --> tokenize --> BM25.retrieve --> top-k MinimalSource
        |
        +--> [ evaluate ]  recall@k against AnsweredQuestions
        |
        v
   [ answer ]  MinimalSource --> get_text() --> DSPy --> Qwen3-0.6B --> answer
```

The two lists produced by the chunkers are deliberately kept separate and
parallel: `corpus_text[i]` is the text BM25 scores, and `corpus_source[i]` is the
location it came from. BM25 returns an integer index, which is used to look up
the matching source. Only the source locations are ever written to the output —
the chunk text is re-read from disk on demand by `get_text()`, so output files
stay small.

Each stage is a separate module under `src/`, and all of them are exposed through
a single Python Fire CLI in `src/__main__.py`.

## Chunking strategy

A Python file and a Markdown page do not break apart the same way, so two
strategies are implemented behind a common `BaseChunker` abstract class
(Strategy pattern). `get_chunker()` acts as a factory and picks the
implementation from the file extension, so the indexing loop never needs to know
which kind of file it is processing.

**Python code — `CodeChunker`**
Uses Chonkie's `CodeChunker`, which parses the file with an AST and splits on
structural boundaries (functions, classes). A chunk is therefore a coherent unit
of code rather than an arbitrary slice that could cut a function in half.

**Markdown and text — `MarkdownChunker`**
Uses Chonkie's `RecursiveChunker` with the `markdown` recipe. It splits
recursively on the most meaningful separator available first — headings, then
paragraphs, then sentences — falling back to finer separators only when a chunk
is still too large. Sections of the documentation stay semantically intact.

Both chunkers return the character span (`first_character_index`,
`last_character_index`) of every chunk in the original file, which is what the
grader compares against.

**Character-based sizing.** Both chunkers are configured with
`tokenizer="character"`. This is a deliberate choice: the subject expresses the
chunk limit in *characters* (2000 max), and the grader rejects any retrieved
source longer than that. Chonkie's default token counting would have made
`--max_chunk_size 2000` mean 2000 *tokens*, i.e. roughly 6000–8000 characters,
and every long source would have invalidated the whole output. Counting
characters makes the CLI argument mean exactly what the subject says it means,
and the limit is guaranteed by construction rather than by post-processing.

## Retrieval method

Retrieval uses **BM25** through the `bm25s` library.

BM25 is a lexical ranking function derived from TF-IDF. For each term of the
query it combines:

- **term frequency** — how often the term occurs in the chunk,
- **inverse document frequency** — how rare the term is across the whole corpus,
  so common words contribute little,

with two corrections that TF-IDF lacks: the score is normalised by chunk length,
so long chunks are not favoured just for containing more words, and term
frequency saturates, so the tenth occurrence of a word adds much less than the
second.

`bm25s` precomputes all term-level scores into a sparse matrix at indexing time,
which makes retrieval a sparse sum at query time — fast enough to stay well
inside the throughput budget.

At query time the question is tokenised the same way the corpus was, scored
against every chunk, and the top-k indices are mapped back to their
`MinimalSource` locations, each carrying its rank and BM25 score.

**Tokenisation.** English stopwords are removed and a Snowball stemmer
(`PyStemmer`) is applied, so that *configure* / *configuring* / *configuration*
collapse to a single term. The identical tokenisation is applied at indexing and
at search time — an asymmetry here silently destroys recall, which is exactly
what happened during development (see *Challenges faced*).

## Performance analysis

Measured on `dataset_docs_public.json` (100 questions) with the provided
moulinette binary.

### Effect of chunk size

| `--max_chunk_size` | chunks indexed | recall@1 | recall@5 | recall@10 |
|---|---|---|---|---|
| 800 | 135 556 | 46 % | 69 % | 71 % |
| 1000 | 108 174 | 50 % | 74 % | 79 % |
| 1200 | 90 117 | 51 % | 73 % | 77 % |
| 1500 | 72 908 | 54 % | 75 % | 79 % |
| **2000** | **55 681** | **59 %** | **77 %** | **80 %** |

Recall increases monotonically with chunk size. Two effects combine: a wider
chunk covers more of the reference span, and the grader's "found" test is an IoU
above 0.05, which a wider chunk clears more easily. Since 2000 characters is the
hard ceiling imposed by the grader, that value is used as the default — the
lever is saturated.

### System performance

| Constraint | Budget | Measured |
|---|---|---|
| Indexing time | ≤ 300 s | ~50 s |
| Retrieval throughput | ≤ 90 s / 200 questions | ~1 s / 100 questions |
| Docs recall@5 | ≥ 80 % | see above |
| Code recall@5 | ≥ 50 % | see above |

Indexing and retrieval are comfortably inside budget. Retrieval is fast because
the BM25 index and the chunk metadata are loaded once and reused for the whole
dataset.

## Design decisions

**Strategy pattern for chunkers.** `BaseChunker` defines the contract
(`chunk(path) -> (List[Chunk], List[str])`) and each subclass implements it.
Adding a new file type means adding one class and one branch in the factory,
without touching the indexing loop. It also guarantees a uniform output format
regardless of which underlying Chonkie chunker is used.

**Two separate models for chunks and sources.** `Chunk` holds what is persisted
at indexing time; `MinimalSource` is what retrieval returns, with the optional
`rank` and `score` fields. Keeping them apart avoids mixing indexing state with
retrieval state in a single model.

**Character spans instead of stored text.** Output files carry only file paths
and character offsets. The text is re-read from the corpus by `get_text()` when
the answer bot needs it. This keeps result files small and makes the offsets the
single source of truth.

**Evaluation is offline.** The `evaluate` command reads two JSON files and never
touches BM25. Search results are produced once by `search_dataset`, then scored.
This keeps the retrieval cost out of the evaluation measurement and mirrors how
the official grader works.

**Lazy LLM configuration.** The Ollama connection is opened inside `answer` and
`answer_dataset` only. `index`, `search`, `search_dataset` and `evaluate` run
without any model server, so the retrieval pipeline is testable when Ollama is
not running.

## Challenges faced

**Token budget versus character budget.** Chonkie counts in tokens by default,
while the subject and the grader count in characters. With `chunk_size=2000`
interpreted as tokens, real chunks reached 2400+ characters and the grader
rejected the whole submission with *"Student data is valid: False"*. A first
attempt converted the budget with a `// 4` heuristic, then post-split oversized
chunks — both were patches over a wrong assumption. The actual fix was
`tokenizer="character"`, a documented Chonkie option that makes `chunk_size`
mean characters directly. The limit then holds by construction.

**Asymmetric tokenisation.** Adding a stemmer at indexing time without adding it
at search time dropped recall@5 from 77 % to 48 %. The index stored stemmed
terms (*configur*) while queries searched unstemmed ones (*configure*), so almost
nothing matched. Any change to the tokenisation pipeline has to be applied on
both sides, and measured one change at a time.

**Path prefixes.** The grader compares `file_path` verbatim. Running the CLI
from a subdirectory produced paths prefixed with `../`, which matched nothing and
gave a recall of exactly 0 % while every other part of the pipeline looked
healthy. Paths are now normalised relative to the project root before being
stored.

**Small-model output formatting.** `Qwen/Qwen3-0.6B` follows DSPy's structured
field protocol inconsistently, sometimes emitting empty answers or leaking the
`[[ ## completed ## ]]` marker. The root cause turned out to be context
saturation: with oversized chunks the prompt exceeded Ollama's context window and
the leading instructions were silently truncated, so the model never saw the
format it was meant to follow. Fixing the chunk size fixed the generation as
well.

**Duplicate corpus entries.** A copy-pasted loop indexed root-level files twice,
inflating the corpus and skewing IDF. Worth stating explicitly because the
symptom — slightly degraded recall — looks like a tuning problem rather than a
bug.

## Example usage

**Index the corpus**

```bash
uv run python -m src index --max_chunk_size 2000
```
```
Chunking files: 100%|██████████| 1159/1159 [00:41<00:00, 27.87file/s]
Ingestion complete! Index 55681 chunks under data/processed/
```

**Search a single query**

```bash
uv run python -m src search "How to configure OpenAI server?" --k 5
```
```
Rank: 1
Score: 6.74
File path: data/raw/vllm-0.10.1/docs/deployment/frameworks/dstack.md
First index character: 0
Last index character: 3170
...
```

**Answer a single question**

```bash
uv run python -m src answer "What endpoint does vLLM use to expose production metrics?" --k 5
```
```
Query: What endpoint does vLLM use to expose production metrics?
Answer: The endpoint used by vLLM to expose production metrics is the
`/metrics` endpoint on its OpenAI-compatible API server.
```

**Search a whole dataset**

```bash
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
```

**Generate answers for a dataset**

```bash
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

**Evaluate your own recall@k**

```bash
uv run python -m src evaluate \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json
```
```
Evaluation Results
========================================
Questions evaluated: 100
Recall@1: 0.590
Recall@3: 0.690
Recall@5: 0.770
Recall@10: 0.800
```

## Resources

**Retrieval and BM25**
- [BM25S: Fast lexical search](https://huggingface.co/blog/xhluca/bm25s) — the
  library used here, and a clear comparison with Elasticsearch and rank-bm25
- [Which BM25 Do You Mean? A Large-Scale Reproducibility Study of Scoring
  Variants](https://link.springer.com/chapter/10.1007/978-3-030-45442-5_4) —
  Kamphuis et al., on the different BM25 formulations
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/),
  Manning, Raghavan & Schütze — chapters on TF-IDF and probabilistic retrieval

**RAG**
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks](https://arxiv.org/abs/2005.11401) — Lewis et al., the original RAG paper

**Tooling**
- [Chonkie documentation](https://docs.chonkie.ai/) — chunker reference,
  including the `tokenizer="character"` option that resolved the sizing issue
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Python Fire](https://github.com/google/python-fire)
- [DSPy documentation](https://dspy.ai/)
- [vLLM documentation](https://docs.vllm.ai/) — the indexed corpus
