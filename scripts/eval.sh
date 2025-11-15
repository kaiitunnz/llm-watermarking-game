#!/bin/bash
# task -> ['qa', 'translation', 'summarization']

set -euo pipefail

uv run -m wmgame --eval --strategy-file $1 --task qa --max-examples 100
uv run -m wmgame --eval --strategy-file $1 --task translation --max-examples 100
uv run -m wmgame --eval --strategy-file $1 --task summarization --max-examples 100
