from typing import Any

from datasets import load_dataset


def split_into_sentences(text: str):
    """
    Very lightweight sentence splitter.
    For proper OpenGen replication you might want spaCy or nltk,
    but this is enough for quick experiments.
    """
    # Normalize whitespace
    text = " ".join(text.split())
    # Naive split on periods
    parts = [s.strip() for s in text.split(".") if s.strip()]
    # Add the period back (so we don't lose punctuation)
    return [p + "." for p in parts]


def load_opengen_qa_from_wikitext(
    max_examples: int = 500,
    prefix_num_sentences: int = 2,
):
    """
    Approximate OpenGen dataset using WikiText-103 validation split.

    Each example:
      - Take first `prefix_num_sentences` sentences as a 'question' context.
      - Treat the remaining text as the long-form 'answer' (gold continuation).

    Returns:
      List of (prompt, target_answer) pairs.
    """
    ds = load_dataset("wikitext", "wikitext-103-v1", split="validation")

    examples = []
    ex: Any
    for ex in ds:
        text = ex["text"].strip()
        if not text:
            continue

        sentences = split_into_sentences(text)
        if len(sentences) <= prefix_num_sentences:
            continue

        prefix = " ".join(sentences[:prefix_num_sentences]).strip()
        continuation = " ".join(sentences[prefix_num_sentences:]).strip()
        if not prefix or not continuation:
            continue

        # Turn the prefix into an open-ended QA style prompt
        prompt = (
            "You are an expert writer.\n"
            "Q: Continue the following passage in a coherent and informative way.\n\n"
            f"{prefix}\n\n"
            "A:"
        )
        target = continuation

        examples.append((prompt, target))
        if len(examples) >= max_examples:
            break

    return examples


def load_translation_examples(
    max_examples: int = 100,
    dataset_name: str = "Helsinki-NLP/opus-100",
    config: str = "en-fr",
):
    """
    Load translation examples from a standard MT dataset.
    Returns (prompt, target) pairs where:
      - prompt: translation instruction
      - target: reference translation
    """
    ds: Any = load_dataset(dataset_name, config, split="train")

    examples = []
    for ex in ds.select(range(min(max_examples, len(ds)))):
        src = ex["translation"]["en"]
        tgt = ex["translation"]["fr"]

        prompt = (
            "You are a high-quality machine translation system.\n"
            "Translate the following sentence from English to French.\n\n"
            f"English: {src}\n\n"
            "French:"
        )
        target = tgt
        examples.append((prompt, target))
    return examples


def load_summarization_examples(
    max_examples: int = 100,
    dataset_name: str = "cnn_dailymail",
    config: str = "3.0.0",
    split: str = "test",
):
    """
    Load summarization examples.
    Returns (prompt, target) pairs where:
      - prompt: summarization instruction
      - target: gold summary
    """
    ds: Any = load_dataset(dataset_name, config, split=split)

    examples = []
    for ex in ds.select(range(min(max_examples, len(ds)))):
        article = ex["article"]
        summary = ex["highlights"]

        prompt = (
            "You are an expert summarizer.\n"
            "Summarize the following news article in a concise paragraph.\n\n"
            f"{article}\n\n"
            "Summary:"
        )
        target = summary
        examples.append((prompt, target))

    return examples
