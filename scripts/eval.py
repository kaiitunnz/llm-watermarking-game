import argparse
import subprocess
from pathlib import Path

STRATEGY_DIR = Path(__file__).resolve().parent.parent / "strategies"
TASK_NAME_MAP = {"qa": "qa", "summ": "summarization", "tl": "translation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy-dir",
        type=Path,
        default=STRATEGY_DIR,
        help="Path to the strategy directory",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum number of examples to evaluate",
    )
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run")
    return parser.parse_args()


def get_cli_command(strategy_file: Path, task: str, max_examples: int) -> list[str]:
    base_command: list[str] = [
        "uv",
        "run",
        "-m",
        "wmgame",
        "--eval",
        "--strategy-file",
        str(strategy_file),
        "--task",
        TASK_NAME_MAP[task],
        "--max-examples",
        str(max_examples),
    ]
    return base_command


def main(strategy_dir: Path, max_examples: int, dry_run: bool) -> None:
    strategy_files = sorted(strategy_dir.glob("*.json"))
    strategy_files = [
        fpath for fpath in strategy_files if "sample" not in fpath.name.lower()
    ]
    num_strategies = len(strategy_files)
    print(f"Found {num_strategies} strategy files.")

    if dry_run:
        print("Dry run mode. The following commands would be executed:")
        for i, strategy_file in enumerate(strategy_files, 1):
            print()
            print(f"Strategy {i}/{num_strategies}: {strategy_file.name}")

            _, task, *_ = strategy_file.stem.split("_")
            task = task.lower()

            if task == "avg":
                for all_task in TASK_NAME_MAP:
                    command = get_cli_command(strategy_file, all_task, max_examples)
                    print(" ".join(command))
            else:
                command = get_cli_command(strategy_file, task, max_examples)
                print(" ".join(command))
        return

    for i, strategy_file in enumerate(strategy_files, 1):
        print(f"Evaluating strategy {i}/{num_strategies}: {strategy_file.name}")

        _, task, *_ = strategy_file.stem.split("_")
        task = task.lower()

        if task == "avg":
            for all_task in TASK_NAME_MAP:
                command = get_cli_command(strategy_file, all_task, max_examples)
                proc = subprocess.run(command)
                if proc.returncode != 0:
                    raise RuntimeError(f"Command failed: {' '.join(command)}")
        else:
            command = get_cli_command(strategy_file, task, max_examples)
            proc = subprocess.run(command)
            if proc.returncode != 0:
                raise RuntimeError(f"Command failed: {' '.join(command)}")


if __name__ == "__main__":
    args = parse_args()
    main(args.strategy_dir, args.max_examples, args.dry_run)
