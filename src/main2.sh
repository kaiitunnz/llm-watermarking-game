#!/bin/bash
# attack -> ['none', 'detection', 'paraphrase', 'frequency']
# task -> ['qa', 'translation', 'summarization']
#!/bin/bash
# attack -> ['none', 'detection', 'paraphrase', 'frequency']
# task -> ['qa', 'translation', 'summarization']
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model exp --attack none --task translation --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model unigram --attack none --task translation --max_examples 100
exit

CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model unigram --attack none --task translation --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model unigram --attack detection --task translation --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model unigram --attack paraphrase --task translation --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model unigram --attack frequency --task translation --max_examples 100

CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model unigram --attack none --task qa --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model unigram --attack detection --task qa --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model unigram --attack paraphrase --task qa --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model unigram --attack frequency --task qa --max_examples 100

CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model unigram --attack none --task summarization --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model unigram --attack detection --task summarization --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model unigram --attack paraphrase --task summarization --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model unigram --attack frequency --task summarization --max_examples 100



CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model exp --attack none --task translation --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model exp --attack detection --task translation --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model exp --attack paraphrase --task translation --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model exp --attack frequency --task translation --max_examples 100

CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model exp --attack none --task qa --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model exp --attack detection --task qa --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model exp --attack paraphrase --task qa --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model exp --attack frequency --task qa --max_examples 100

CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model exp --attack none --task summarization --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model exp --attack detection --task summarization --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model exp --attack paraphrase --task summarization --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack.py --wm_model exp --attack frequency --task summarization --max_examples 100