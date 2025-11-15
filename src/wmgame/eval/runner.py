import logging
import time
from collections import defaultdict
from pathlib import Path

import evaluate

from wmgame.attack.base import ResultEntry, log_result, log_summary, write_result
from wmgame.eval.base import (
    SUPPORTED_ATTACKS,
    BaseRunner,
    StrategySampler,
    seed_everything,
)
from wmgame.eval.exp import ExpRunner
from wmgame.eval.kgw import KGWRunner
from wmgame.eval.unigram import UnigramRunner
from wmgame.tasks import (
    load_opengen_qa_from_wikitext,
    load_summarization_examples,
    load_translation_examples,
)


SCHEME_FACTORIES: dict[str, type[BaseRunner]] = {
    KGWRunner.name: KGWRunner,
    UnigramRunner.name: UnigramRunner,
    ExpRunner.name: ExpRunner,
}


def _load_examples(task: str, max_examples: int) -> list[tuple[str, str]]:
    if task == "qa":
        return load_opengen_qa_from_wikitext(max_examples=max_examples)
    if task == "translation":
        return load_translation_examples(max_examples=max_examples)
    if task == "summarization":
        return load_summarization_examples(max_examples=max_examples)
    raise ValueError(f"Unknown task: {task}")


def evaluate_mixed_strategy(
    *,
    strategy: StrategySampler,
    task: str,
    max_examples: int,
    model_name: str,
    result_file: Path,
    logger: logging.Logger,
    seed: int,
) -> None:
    rng = seed_everything(seed)
    comet = evaluate.load("comet")
    bertscore = evaluate.load("bertscore")
    examples = _load_examples(task, max_examples)

    runners: dict[str, BaseRunner] = {}
    for watermark in strategy.watermark_actions:
        factory = SCHEME_FACTORIES.get(watermark)
        if factory is None:
            raise ValueError(f"Unsupported watermark scheme '{watermark}'")
        runners[watermark] = factory(model_name)

    pair_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0, "success": 0, "task_score": 0.0, "elapsed": 0.0}
    )
    total_examples = 0
    successful_attacks = 0
    total_task_score = 0.0
    total_time = 0.0
    task_metric = "COMET score" if task == "translation" else "BERTScore F1"

    for idx, (prompt, target) in enumerate(examples):
        watermark, attack_method = strategy.sample(rng)
        if attack_method not in SUPPORTED_ATTACKS:
            raise ValueError(f"Unsupported attack '{attack_method}' in strategy")
        runner = runners[watermark]
        start_time = time.time()
        output_tokens = runner.generate(attack_method, prompt)
        generated_text = runner.decode(output_tokens)

        if task == "translation":
            comet_output = comet.compute(
                predictions=[generated_text],
                references=[target],
                sources=[prompt],
            )
            assert comet_output is not None
            score = float(comet_output["scores"][0])
        else:
            bert_res = bertscore.compute(
                predictions=[generated_text],
                references=[target],
                model_type="microsoft/deberta-xlarge-mnli",
            )
            assert bert_res is not None
            score = float(bert_res["f1"][0])

        detection_result = runner.detect(output_tokens, generated_text)
        elapsed_time = time.time() - start_time

        total_examples += 1
        total_task_score += score
        total_time += elapsed_time
        if not detection_result.passed:
            successful_attacks += 1
        pair_key = (watermark, attack_method)
        stats = pair_stats[pair_key]
        stats["count"] += 1
        stats["task_score"] += score
        stats["elapsed"] += elapsed_time
        if not detection_result.passed:
            stats["success"] += 1

        result = ResultEntry(
            sample_idx=idx,
            prompt=prompt,
            target=target,
            output=generated_text,
            task_score=score,
            z_score=detection_result.z_score,
            p_value=detection_result.p_value,
            passed=detection_result.passed,
            elapsed_time=elapsed_time,
            watermark=watermark,
            attack_method=attack_method,
        )
        write_result(result_file, result)
        log_result(logger, result)

    if total_examples == 0:
        logger.info("No examples evaluated.")
        return

    log_summary(
        logger,
        attack_method=f"mixed:{strategy.description}",
        total_examples=total_examples,
        asr=successful_attacks / total_examples,
        task_metric=task_metric,
        task_score=total_task_score / total_examples,
        total_time=total_time,
    )
    logger.info("Breakdown by (watermark, attack):")
    for (watermark, attack), stats in sorted(pair_stats.items()):
        count = int(stats["count"])
        if count == 0:
            continue
        asr = stats["success"] / count
        avg_score = stats["task_score"] / count
        avg_time = stats["elapsed"] / count
        logger.info(
            "  %s | %s -> n=%d, ASR=%.2f%%, avg score=%.4f, avg time=%.2fs",
            watermark,
            attack,
            count,
            asr * 100.0,
            avg_score,
            avg_time,
        )
