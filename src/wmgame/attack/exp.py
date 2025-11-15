import time
import logging
from pathlib import Path

import evaluate

from wmgame.attack.base import ResultEntry, log_result, log_summary, write_result
from wmgame.tasks import (
    load_opengen_qa_from_wikitext,
    load_summarization_examples,
    load_translation_examples,
)
from wmgame.watermark.base import DetectionConfig
from wmgame.watermark.exp import ExpWatermarkedLLM, ExpDetector


def attack_exp(
    model_name: str,
    attack_method: str,
    task: str,
    max_examples: int,
    result_file: str | Path,
    logger: logging.Logger,
):
    result_file = Path(result_file)

    # Initialize metrics
    comet = evaluate.load("comet")
    bertscore = evaluate.load("bertscore")
    total_examples = 0
    successful_attacks = 0
    total_task_score = 0.0
    total_time = 0.0

    # Default configurations
    max_new_tokens = 128
    k = max_new_tokens
    n = 256
    seed = 0
    gamma = 0.3
    detection_threshold = 0.05

    # Load examples based on task
    if task == "qa":
        all_prompts = load_opengen_qa_from_wikitext(max_examples=max_examples)
    elif task == "translation":
        all_prompts = load_translation_examples(max_examples=max_examples)
    elif task == "summarization":
        all_prompts = load_summarization_examples(max_examples=max_examples)
    else:
        raise ValueError(f"Unknown task: {task}")

    # Initialize LLM and detector
    llm = ExpWatermarkedLLM(model_name)
    detector = ExpDetector(
        tokenizer=llm.tokenizer,
        n=n,
        k=k,
        gamma=gamma,
        seed=seed,
        vocab_size=len(llm.tokenizer),
        config=DetectionConfig(threshold=detection_threshold),
    )

    for i, (prompt, target) in enumerate(all_prompts):
        start_time = time.time()

        if attack_method == "detection":
            output = llm.generate_with_detection_attack(
                detector=detector,
                prompt=prompt,
                n=n,
                seed=seed,
                num_samples=10,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        elif attack_method == "frequency":
            output = llm.generate_with_frequency_attack(
                prompt=prompt,
                n=n,
                seed=seed,
                num_samples=10,
                reduction_factor=0.1,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        elif attack_method == "paraphrase":
            output = llm.generate_with_paraphrase_attack(
                prompt=prompt,
                n=n,
                seed=seed,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        elif attack_method == "translation":
            output = llm.generate_with_translation_attack(
                prompt=prompt,
                n=n,
                seed=seed,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        elif attack_method == "none":
            output = llm.generate_with_watermark(
                prompt=prompt,
                n=n,
                seed=seed,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")

        generated_tokens = output[0].cpu()
        generated_text = llm.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        if task == "translation":
            comet_output = comet.compute(
                predictions=[generated_text],
                references=[target],
                sources=[prompt],  # COMET needs source text for translation
            )
            assert comet_output is not None
            score = comet_output["scores"][0]
            total_task_score += score
        else:
            # Calculate BERTScore F1
            bert_res = bertscore.compute(
                predictions=[generated_text],
                references=[target],
                model_type="microsoft/deberta-xlarge-mnli",  # or "bert-base-uncased"
            )
            assert bert_res is not None
            score = float(bert_res["f1"][0])
            total_task_score += score

        # Detect watermark
        detection_result = detector.detect_tokens(generated_tokens)

        # Track attack success
        total_examples += 1
        if not detection_result.passed:
            successful_attacks += 1

        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        result = ResultEntry(
            sample_idx=i,
            prompt=prompt,
            target=target,
            output=generated_text,
            task_score=score,
            z_score=detection_result.z_score,
            p_value=detection_result.p_value,
            passed=detection_result.passed,
            elapsed_time=elapsed_time,
        )
        write_result(result_file, result)
        log_result(logger, result)

    log_summary(
        logger,
        attack_method,
        total_examples,
        successful_attacks / total_examples,
        "COMET score" if task == "translation" else "BERTScore F1",
        total_task_score / total_examples,
        total_time,
    )
