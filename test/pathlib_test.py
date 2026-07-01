import pathlib

dir_name = ["vllm", "docs"]
extensions = ["*.py", "*.md", "*.txt",]
vllm_path = pathlib.Path("data/raw/vllm-0.10.1")
all_path = []
for dir in dir_name:
    actual_path = vllm_path / dir
    for ext in extensions:
        all_path.extend(list(actual_path.rglob(ext)))
for ext in extensions:
    all_path.extend(list(vllm_path.glob(ext)))

print(all_path)
