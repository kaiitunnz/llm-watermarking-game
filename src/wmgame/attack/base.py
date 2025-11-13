import json
import logging
from pathlib import Path
from typing import TypedDict


class ResultEntry(TypedDict):
    sample_idx: int
    prompt: str
    target: str
    output: str
    task_score: float
    z_score: float | None
    p_value: float | None
    passed: bool | None
    elapsed_time: float


def write_result(result_file: Path, result: ResultEntry) -> None:
    with result_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def log_result(logger: logging.Logger, result: ResultEntry) -> None:
    logger.info("-" * 60)
    logger.info("Example %d", result["sample_idx"] + 1)
    logger.info("-" * 60)
    logger.info("Prompt: %r", result["prompt"])
    logger.info("Target: %r", result["target"])
    logger.info("Generated text: %r", result["output"])
    if result["z_score"] is not None:
        logger.info("Z-score: %s", result["z_score"])
    if result["p_value"] is not None:
        logger.info("P-value: %s", result["p_value"])
    logger.info("Passed detection: %s", result["passed"])
    logger.info("Task score: %s", result["task_score"])
    logger.info("Elapsed time: %s s", result["elapsed_time"])


def log_summary(
    logger: logging.Logger,
    attack_method: str,
    total_examples: int,
    asr: float,
    task_metric: str,
    task_score: float,
    total_time: float,
) -> None:
    logger.info("-" * 60)
    logger.info("Attack Summary")
    logger.info("-" * 60)
    logger.info("Attack method: %s", attack_method)
    logger.info("Total examples: %d", total_examples)
    logger.info("Attack success rate: %f", asr * 100)
    logger.info("Average %s: %f", task_metric, task_score)
    logger.info("Total execution time: %f s", total_time)
