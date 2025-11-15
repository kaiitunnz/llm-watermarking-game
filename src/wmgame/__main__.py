import os
import logging
from argparse import ArgumentParser
from pathlib import Path


parser = ArgumentParser()
parser.add_argument(
    "--watermark",
    type=str,
    default="kgw",
    choices=["kgw", "unigram", "exp"],
)
parser.add_argument(
    "--attack",
    type=str,
    default="none",
    choices=["none", "detection", "paraphrase", "frequency", "translation"],
    help="Attack method to use",
)
parser.add_argument(
    "--task",
    type=str,
    default="qa",
    choices=["qa", "translation", "summarization"],
    help="Task to evaluate on",
)
parser.add_argument(
    "--max-examples",
    type=int,
    default=100,
    help="Maximum number of examples to process",
)
parser.add_argument(
    "--model-name",
    type=str,
    default="meta-llama/Llama-3.1-8B-Instruct",
    help="Name of the model to use",
)
parser.add_argument(
    "--log-dir",
    type=str,
    default="outputs",
    help="Directory to save logs and results",
)
parser.add_argument(
    "--eval",
    action="store_true",
    help="Run mixed-strategy evaluation using a strategy file",
)
parser.add_argument(
    "--strategy-file",
    type=str,
    help="Path to a JSON file containing the mixed strategy (required for --eval)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed used for sampling mixed strategies",
)
args = parser.parse_args()

log_dir = Path(args.log_dir)
os.makedirs(log_dir, exist_ok=True)

if args.eval:
    if not args.strategy_file:
        parser.error("--strategy-file is required when using --eval")
    strategy_path = Path(args.strategy_file)
    strategy_label = strategy_path.stem
    log_filename = log_dir / f"eval_results_{strategy_label}_{args.task}.log"
    result_filename = log_dir / f"eval_data_{strategy_label}_{args.task}.jsonl"
else:
    log_filename = (
        log_dir / f"attack_results_{args.watermark}_{args.attack}_{args.task}.log"
    )
    result_filename = (
        log_dir / f"attack_data_{args.watermark}_{args.attack}_{args.task}.jsonl"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, mode="w"),
    ],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

result_filename.unlink(missing_ok=True)

if args.eval:
    from wmgame.eval import StrategySampler, evaluate_mixed_strategy

    strategy = StrategySampler.from_path(strategy_path)  # type: ignore
    evaluate_mixed_strategy(
        strategy=strategy,
        task=args.task,
        max_examples=args.max_examples,
        model_name=args.model_name,
        result_file=result_filename,
        logger=logger,
        seed=args.seed,
    )
else:
    if args.watermark == "kgw":
        from wmgame.attack import attack_kgw

        attack_kgw(
            model_name=args.model_name,
            attack_method=args.attack,
            task=args.task,
            max_examples=args.max_examples,
            result_file=result_filename,
            logger=logger,
        )
    elif args.watermark == "unigram":
        from wmgame.attack import attack_unigram

        attack_unigram(
            model_name=args.model_name,
            attack_method=args.attack,
            task=args.task,
            max_examples=args.max_examples,
            result_file=result_filename,
            logger=logger,
        )
    elif args.watermark == "exp":
        from wmgame.attack import attack_exp

        attack_exp(
            model_name=args.model_name,
            attack_method=args.attack,
            task=args.task,
            max_examples=args.max_examples,
            result_file=result_filename,
            logger=logger,
        )
    else:
        raise ValueError(f"Unknown watermark method: {args.watermark}")
