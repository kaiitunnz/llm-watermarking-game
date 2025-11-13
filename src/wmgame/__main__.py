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
    choices=["none", "detection", "paraphrase", "frequency"],
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
args = parser.parse_args()

# Set up logging
log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(
    log_dir, f"attack_results_{args.watermark}_{args.attack}_{args.task}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler(log_filename, mode="w"),  # Output to file
    ],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

result_filename = Path(
    log_dir, f"attack_data_{args.watermark}_{args.attack}_{args.task}.jsonl"
)
result_filename.unlink(missing_ok=True)

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
        logger=logger,
    )
elif args.watermark == "exp":
    from wmgame.attack import attack_exp

    attack_exp(
        model_name=args.model_name,
        attack_method=args.attack,
        task=args.task,
        max_examples=args.max_examples,
        logger=logger,
    )
else:
    raise ValueError(f"Unknown watermark method: {args.watermark}")
