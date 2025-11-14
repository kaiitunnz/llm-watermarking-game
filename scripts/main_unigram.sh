#!/bin/bash
# attack -> ['none', 'detection', 'paraphrase', 'frequency']
# task -> ['qa', 'translation', 'summarization']

set -euo pipefail

uv run -m wmgame --watermark unigram --attack none --task translation --max-examples 100
uv run -m wmgame --watermark unigram --attack detection --task translation --max-examples 100
uv run -m wmgame --watermark unigram --attack paraphrase --task translation --max-examples 100
uv run -m wmgame --watermark unigram --attack frequency --task translation --max-examples 100
                                       
uv run -m wmgame --watermark unigram --attack none --task qa --max-examples 100
uv run -m wmgame --watermark unigram --attack detection --task qa --max-examples 100
uv run -m wmgame --watermark unigram --attack paraphrase --task qa --max-examples 100
uv run -m wmgame --watermark unigram --attack frequency --task qa --max-examples 100
                                       
uv run -m wmgame --watermark unigram --attack none --task summarization --max-examples 100
uv run -m wmgame --watermark unigram --attack detection --task summarization --max-examples 100
uv run -m wmgame --watermark unigram --attack paraphrase --task summarization --max-examples 100
uv run -m wmgame --watermark unigram --attack frequency --task summarization --max-examples 100