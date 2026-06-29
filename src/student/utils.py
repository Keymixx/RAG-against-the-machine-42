def get_text(file_path: str, first_index: int, last_index: int) -> str:
    with open(file_path, "r") as f:
        text = f.read()
        return text[first_index:last_index]