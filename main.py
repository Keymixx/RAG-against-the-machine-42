# from bm25s import BM25, tokenize
# import dspy
# import pathlib
# from student.chunkers import get_chunker
# from src.answer import AnswerBot
# from student.models import MinimalSource
# from transformers import AutoTokenizer
# from typing import List

# max_token_size = 256
# model = "Qwen/Qwen3-0.6B"
# tokenizer = AutoTokenizer.from_pretrained(model)

# corpus_text = []
# corpus_source = []
# all_path = []

# path = pathlib.Path("vllm-0.10.1")
# doc_path = pathlib.Path("vllm-0.10.1/docs")

# all_path.extend(list(doc_path.rglob("*.md")))
# all_path.extend(list(doc_path.rglob("vllm/*.py")))
# all_path.extend(list(path.glob("README.md")))
# # all_path.extend(list(path.rglob("*.py")))
# for path in all_path:
#     chunker = get_chunker(path, tokenizer, max_token_size)
#     chunks = chunker.chunk(path)
#     for chunk in chunks:
#         corpus_text.append(chunk.text)
#         corpus_source.append(chunk)

# # print(corpus_text[0])
# # print(corpus_text[1])
# retriever = BM25()
# retriever.index(tokenize(corpus_text, stopwords="english"))
# retriever.save("bm25s_index_vllm")

# query = "What endpoint does vLLM use to expose production metrics?"
# results, scores = retriever.retrieve(tokenize(query, stopwords="english"), k=2)


# lm = dspy.LM(
#     'ollama_chat/qwen3:0.6b',
#     api_base='http://localhost:11434', #CHANGE CA !!!!
#     max_tokens=max_token_size,
#     think=False
# )
# dspy.configure(lm=lm)


# module = AnswerBot(max_token_size)
# sources: List[MinimalSource] = []

# for i in range(results.shape[1]):
#     doc_index = results[0, i]
#     score = scores[0, i]
#     print("\n\n")
#     print(f"Rank {i+1} (score: {score:.2f})")
#     print(f"Source index : {doc_index}")
#     print(f"Source file  : {corpus_source[doc_index].file_path}")
#     print(f"First index  : {corpus_source[doc_index].first_character_index}")
#     print(f"Last index  : {corpus_source[doc_index].last_character_index}")
#     sources.append(corpus_source[doc_index])

# response = module(query, sources)
# print(response.answer)

# from src.student import indexing

# indexing(256)
import json

with open("data/processed/chunks/chunks.json", "r") as f:
    chunks = json.load(f)

for c in chunks:
    print(c)
    print("\n\n\n\n\n\n\n\n\\n\n\n")
