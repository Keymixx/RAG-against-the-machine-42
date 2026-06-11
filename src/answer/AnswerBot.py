import dspy
from src.models import MinimalSource
from typing import List

class AnswerSignature(dspy.Signature):
    question: str = dspy.InputField()
    sources: List[MinimalSource] = dspy.InputField()
    answer: str = dspy.OutputField()


class AnswerBot(dspy.Module):
    def __init__(self, max_tokens: int):
        super().__init__()		
        self.prog = dspy.Predict(AnswerSignature)
    
    def forward(self, question: str, sources: List[MinimalSource]):
        return self.prog(question=question, sources=sources)