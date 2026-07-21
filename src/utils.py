def get_text(file_path: str, first_index: int, last_index: int) -> str:
    """Read a slice of a text file.

    Args:
        file_path: Path to the file to read.
        first_index: Index (inclusive) of the first character to return.
        last_index: Index (exclusive) of the last character to return.

    Returns:
        The substring text[first_index:last_index] of the file content.

    Raises:
        OSError: If the file cannot be found or read.
    """
    try:
        with open(file_path, "r") as f:
            text = f.read()
            return text[first_index:last_index]
    except OSError as exc:
        raise OSError(f"get_text: unable to read {file_path}: {exc}") from exc
