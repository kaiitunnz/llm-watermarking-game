#!/bin/bash
# attack -> ['none', 'detection', 'paraphrase', 'frequency', "translation"]
# task -> ['qa', 'translation', 'summarization']

set -euo pipefail

uv run -m wmgame --watermark kgw --attack none --task translation --max-examples 100
uv run -m wmgame --watermark kgw --attack detection --task translation --max-examples 100
uv run -m wmgame --watermark kgw --attack paraphrase --task translation --max-examples 100
uv run -m wmgame --watermark kgw --attack frequency --task translation --max-examples 100
uv run -m wmgame --watermark kgw --attack translation --task translation --max-examples 100
                            
uv run -m wmgame --watermark kgw --attack none --task qa --max-examples 100
uv run -m wmgame --watermark kgw --attack detection --task qa --max-examples 100
uv run -m wmgame --watermark kgw --attack paraphrase --task qa --max-examples 100
uv run -m wmgame --watermark kgw --attack frequency --task qa --max-examples 100
uv run -m wmgame --watermark kgw --attack translation --task qa --max-examples 100
                            
uv run -m wmgame --watermark kgw --attack none --task summarization --max-examples 100
uv run -m wmgame --watermark kgw --attack detection --task summarization --max-examples 100
uv run -m wmgame --watermark kgw --attack paraphrase --task summarization --max-examples 100
uv run -m wmgame --watermark kgw --attack frequency --task summarization --max-examples 100
uv run -m wmgame --watermark kgw --attack translation --task summarization --max-examples 100
