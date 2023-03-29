"""
    utils.py - Utility functions for the project.
"""

import re
from pathlib import Path

from natsort import natsorted


def truncate_word_count(text, max_words=512):
    """
    truncate_word_count - a helper function for the gradio module
    Parameters
    ----------
    text : str, required, the text to be processed
    max_words : int, optional, the maximum number of words, default=512
    Returns
    -------
    dict, the text and whether it was truncated
    """
    # split on whitespace with regex
    words = re.split(r"\s+", text)
    processed = {}
    if len(words) > max_words:
        processed["was_truncated"] = True
        processed["truncated_text"] = " ".join(words[:max_words])
    else:
        processed["was_truncated"] = False
        processed["truncated_text"] = text
    return processed


def load_examples(src):
    """
    load_examples - a helper function for the gradio module to load examples
    Returns:
        list of str, the examples
    """
    src = Path(src)
    src.mkdir(exist_ok=True)
    examples = [f for f in src.glob("*.txt")]
    examples = natsorted(examples)
    # load the examples into a list
    text_examples = []
    for example in examples:
        with open(example, "r") as f:
            text = f.read()
            text_examples.append([text, "large", 2, 512, 0.7, 3.5, 3])

    return text_examples


def load_example_filenames(example_path: str or Path):
    """
    load_example_filenames - a helper function for the gradio module to load examples
    Returns:
        dict, the examples (filename:full path)
    """
    example_path = Path(example_path)
    # load the examples into a list
    examples = {f.name: f for f in example_path.glob("*.txt")}
    return examples