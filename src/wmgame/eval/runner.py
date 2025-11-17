import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import evaluate
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)
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


class CachedResultStore:
    def __init__(self, base_dir: Path, task: str) -> None:
        self.base_dir = base_dir
        self.task = task
        self._cache: dict[tuple[str, str], dict[int, dict]] = {}

    def _ensure_loaded(self, watermark: str, attack: str) -> None:
        key = (watermark, attack)
        if key in self._cache:
            return
        path = self.base_dir / f"attack_data_{watermark}_{attack}_{self.task}.jsonl"
        entries: dict[int, dict] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    idx = int(entry.get("sample_idx", len(entries)))
                    entries[idx] = entry
        self._cache[key] = entries

    def has(self, watermark: str, attack: str, sample_idx: int) -> bool:
        self._ensure_loaded(watermark, attack)
        return sample_idx in self._cache[(watermark, attack)]

    def get(self, watermark: str, attack: str, sample_idx: int) -> dict:
        self._ensure_loaded(watermark, attack)
        return self._cache[(watermark, attack)][sample_idx]


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
    use_dummy_model: bool = True,  # Load all results from cache when True
) -> None:
    rng = seed_everything(seed)
    comet = evaluate.load("comet")
    bertscore = evaluate.load("bertscore")
    examples = _load_examples(task, max_examples)
    cache_store = CachedResultStore(result_file.parent, task)
    baseline_result_file = result_file.with_name(f"{result_file.stem}_baseline.jsonl")
    baseline_result_file.unlink(missing_ok=True)

    if use_dummy_model:
        model_or_name = "dummy"
    else:
        # Create shared base model and tokenizer
        if TYPE_CHECKING:
            base_model = LlamaForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )
            base_tokenizer = LlamaTokenizer.from_pretrained(
                model_name, torch_dtype=torch.float16
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )
            base_tokenizer = AutoTokenizer.from_pretrained(
                model_name, torch_dtype=torch.float16
            )
        model_or_name = (base_model, base_tokenizer)

    runners: dict[str, BaseRunner] = {}
    for watermark in strategy.watermark_actions:
        factory = SCHEME_FACTORIES.get(watermark)
        if factory is None:
            raise ValueError(f"Unsupported watermark scheme '{watermark}'")
        runners[watermark] = factory(model_or_name)

    def infer_passed(
        watermark: str, z_score: float | None, p_value: float | None
    ) -> bool | None:
        runner = runners[watermark]
        threshold = runner.detection_threshold
        if watermark == "exp":
            if p_value is None:
                return None
            return float(p_value) <= threshold
        elif watermark in ["kgw", "unigram"]:
            if z_score is None:
                return None
            return float(z_score) >= threshold

    pair_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0, "success": 0, "task_score": 0.0, "elapsed": 0.0}
    )
    baseline_stats: dict[str, dict[str, float]] = defaultdict(
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
        use_cache = cache_store.has(watermark, attack_method, idx)
        if use_cache:
            cached = cache_store.get(watermark, attack_method, idx)
            generated_text = cached["output"]
            prompt_used = cached.get("prompt", prompt)
            target_used = cached.get("target", target)
            score = float(cached.get("task_score", 0.0))
            elapsed_time = float(cached.get("elapsed_time", 0.0))
            z_score = cached.get("z_score")
            p_value = cached.get("p_value")
            passed = infer_passed(watermark, z_score, p_value)
        else:
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
            z_score = detection_result.z_score
            p_value = detection_result.p_value
            passed = detection_result.passed
            prompt_used = prompt
            target_used = target

        total_examples += 1
        total_task_score += score
        total_time += elapsed_time
        if passed is False:
            successful_attacks += 1
        pair_key = (watermark, attack_method)
        stats = pair_stats[pair_key]
        stats["count"] += 1
        stats["task_score"] += score
        stats["elapsed"] += elapsed_time
        if passed is False:
            stats["success"] += 1

        # Update defender baseline using cached attack=none results when available
        if cache_store.has(watermark, "none", idx):
            baseline_entry = cache_store.get(watermark, "none", idx)
            baseline_data = baseline_stats[watermark]
            baseline_data["count"] += 1
            baseline_data["task_score"] += float(baseline_entry.get("task_score", 0.0))
            baseline_data["elapsed"] += float(baseline_entry.get("elapsed_time", 0.0))
            baseline_passed = infer_passed(
                watermark, baseline_entry.get("z_score"), baseline_entry.get("p_value")
            )
            if baseline_passed is False:
                baseline_data["success"] += 1
            baseline_result = ResultEntry(
                sample_idx=int(baseline_entry.get("sample_idx", idx)),
                prompt=baseline_entry.get("prompt", prompt),
                target=baseline_entry.get("target", target),
                output=baseline_entry.get("output", ""),
                task_score=float(baseline_entry.get("task_score", 0.0)),
                z_score=baseline_entry.get("z_score"),
                p_value=baseline_entry.get("p_value"),
                passed=baseline_passed,
                elapsed_time=float(baseline_entry.get("elapsed_time", 0.0)),
                watermark=watermark,
                attack_method="none",
            )
            write_result(baseline_result_file, baseline_result)

        result = ResultEntry(
            sample_idx=idx,
            prompt=prompt_used,
            target=target_used,
            output=generated_text,
            task_score=score,
            z_score=z_score,
            p_value=p_value,
            passed=passed,
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

    baseline_logged = False
    combined_baseline = {"count": 0, "success": 0, "task_score": 0.0, "elapsed": 0.0}
    for watermark in sorted(baseline_stats.keys()):
        stats = baseline_stats[watermark]
        count = int(stats["count"])
        if count == 0:
            continue
        if not baseline_logged:
            logger.info("Defender payoff (attack=none baseline):")
            baseline_logged = True
        asr = stats["success"] / count
        avg_score = stats["task_score"] / count
        total_time_baseline = stats["elapsed"]
        logger.info(
            "  %s -> ASR=%.2f%%, avg score=%.4f, total time=%.2fs",
            watermark,
            asr * 100.0,
            avg_score,
            total_time_baseline,
        )
        combined_baseline["count"] += count
        combined_baseline["success"] += stats["success"]
        combined_baseline["task_score"] += stats["task_score"]
        combined_baseline["elapsed"] += stats["elapsed"]
    if baseline_logged and combined_baseline["count"] > 0:
        log_summary(
            logger,
            attack_method="baseline:defender",
            total_examples=int(combined_baseline["count"]),
            asr=combined_baseline["success"] / combined_baseline["count"],
            task_metric=task_metric,
            task_score=combined_baseline["task_score"] / combined_baseline["count"],
            total_time=combined_baseline["elapsed"],
        )
