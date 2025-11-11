#!/bin/bash
# attack -> ['none', 'detection', 'paraphrase', 'frequency']
# task -> ['qa', 'translation', 'summarization']
#!/bin/bash
# attack -> ['none', 'detection', 'paraphrase', 'frequency']
# task -> ['qa', 'translation', 'summarization']

CUDA_VISIBLE_DEVICES=0,1 uv run -m wmgame --watermark exp --attack none --task translation --max-examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run -m wmgame --watermark exp --attack detection --task translation --max-examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run -m wmgame --watermark exp --attack paraphrase --task translation --max-examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run -m wmgame --watermark exp --attack frequency --task translation --max-examples 100

CUDA_VISIBLE_DEVICES=0,1 uv run -m wmgame --watermark exp --attack none --task qa --max-examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run -m wmgame --watermark exp --attack detection --task qa --max-examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run -m wmgame --watermark exp --attack paraphrase --task qa --max-examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run -m wmgame --watermark exp --attack frequency --task qa --max-examples 100

CUDA_VISIBLE_DEVICES=0,1 uv run -m wmgame --watermark exp --attack none --task summarization --max-examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run -m wmgame --watermark exp --attack detection --task summarization --max-examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run -m wmgame --watermark exp --attack paraphrase --task summarization --max-examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run -m wmgame --watermark exp --attack frequency --task summarization --max-examples 100