
import dspy

# Pass the key explicitly...
lm = dspy.LM(
    'ollama_chat/qwen3:0.6b',
    api_base='http://localhost:11434',
    max_tokens=512,
    think=False
)

dspy.configure(lm=lm)

haiku_signature = "subject -> haiku"
haiku_generator = dspy.Predict(haiku_signature)
result = haiku_generator(subject="computer science")
print(result.haiku)
