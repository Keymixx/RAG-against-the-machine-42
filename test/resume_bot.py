class ResumeSignature(dspy.Signature):
    source: str = dspy.InputField(desc="source of doc")
    resume: str = dspy.OutputField(desc="short resume in one sentence")


class ResumeBot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(ResumeSignature)

    def forward(self, source: str) -> str:
        return self.prog(source=source)
