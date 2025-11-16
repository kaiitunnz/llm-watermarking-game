See [base.py](base.py) for the interfaces of watermarked LLMs and detectors.

## WatermarkedLLM

- `generate`: One-shot _unwatermarked_ generation.
- `generate_with_watermark`: One-shot _watermarked_ generation.
- `generation_context`: Create a generation context for token-by-token generation.

## GenerationContext

- `step`: Output the next _unwatermarked_ logits.
- `step_with_watermark`: Output the next _watermarked_ logits.
- `set_next_token`: Set the next token sampled from the logits. Must be called after stepping.

## WatermarkDetector

- `detect`: Detect whether the input text is watermarked.

There are three watermarking schemes: Exp, KGW, and Unigram. See the corresponding modules for usage.
