import time
import logging

import evaluate

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
    logger: logging.Logger,
):
    start_time = time.time()
    logger.setLevel(logging.INFO)
    # Initialize metrics
    comet = evaluate.load("comet")
    bertscore = evaluate.load("bertscore")
    total_examples = 0
    successful_attacks = 0
    total_comet = 0.0
    total_bert_f1 = 0.0
    total_generation_time = 0.0

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
        n=256,
        k=1,
        gamma=0.3,
        seed=0,
        vocab_size=llm.tokenizer.vocab_size,
        config=DetectionConfig(threshold=0.7),
    )

    for prompt, target in all_prompts:
        example_start_time = time.time()

        if attack_method == "detection":
            output = llm.generate_with_detection_attack(
                prompt=prompt,
                gamma=0.3,
                delta=1.0,
                n=32,
                m=32,
                seed=0,
                max_new_tokens=32,
                do_sample=False,
            )
        elif attack_method == "frequency":
            output = llm.generate_with_frequency_attack(
                prompt=prompt,
                gamma=0.3,
                delta=1.0,
                n=32,
                m=32,
                seed=0,
                max_new_tokens=32,
                do_sample=False,
            )
        elif attack_method == "paraphrase":
            output = llm.generate_with_paraphrase_attack(
                prompt=prompt,
                gamma=0.3,
                delta=1.0,
                n=32,
                m=32,
                seed=0,
                max_new_tokens=32,
                do_sample=False,
            )
        elif attack_method == "none":
            output = llm.generate_with_watermark(
                prompt=prompt,
                gamma=0.3,
                delta=1.0,
                n=32,
                m=32,
                seed=0,
                max_new_tokens=32,
                do_sample=False,
            )
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")

        generated_text = llm.tokenizer.decode(output[0], skip_special_tokens=True)

        if task == "translation":
            comet_output = comet.compute(
                predictions=[generated_text],
                references=[target],
                sources=[prompt],  # COMET needs source text for translation
            )
            assert comet_output is not None
            score = comet_output["scores"][0]
            total_comet += score
        else:
            # Calculate BERTScore F1
            bert_res = bertscore.compute(
                predictions=[generated_text],
                references=[target],
                model_type="microsoft/deberta-xlarge-mnli",  # or "bert-base-uncased"
            )
            assert bert_res is not None
            score = float(bert_res["f1"][0])
            total_bert_f1 += score

        logger.info(f"Target: {target}")
        logger.info(f"Generated text: {generated_text}")

        # Detect watermark
        result = detector.detect(generated_text)
        logger.info(f"Detection result: p={result.p_value:.3f}, passed={result.passed}")
        if task == "translation":
            logger.info(f"COMET score: {score:.3f}")
        else:
            logger.info(f"BERTScore F1: {score:.3f}")

        # Track attack success
        total_examples += 1
        if not result.passed:
            successful_attacks += 1

        # logger.info("-" * 60)
        example_time = time.time() - example_start_time
        total_generation_time += example_time
        logger.info(f"One step executing time: {example_time:.3f} s")

    total_time = time.time() - start_time

    # Print summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("ATTACK SUMMARY")
    logger.info("-" * 60)
    logger.info(f"Attack method: {attack_method}")
    logger.info(f"Total examples: {total_examples}")
    logger.info(f"Attack success rate: {(successful_attacks/total_examples)*100:.2f}%")
    if task == "translation":
        logger.info(f"Average COMET score: {(total_comet/total_examples):.3f}")
    else:
        logger.info(f"Average BERTScore F1: {(total_bert_f1/total_examples):.3f}")
    logger.info(f"Total execution time: {total_time:.2f}s")
    logger.info("=" * 60)
