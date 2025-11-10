#!/bin/bash
# attack -> ['none', 'detection', 'paraphrase', 'frequency']
# task -> ['qa', 'translation', 'summarization']
#!/bin/bash
# attack -> ['none', 'detection', 'paraphrase', 'frequency']
# task -> ['qa', 'translation', 'summarization']

CUDA_VISIBLE_DEVICES=2,3 uv run attack_exp.py --wm_model exp --attack none --task translation --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack_exp.py --wm_model exp --attack detection --task translation --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack_exp.py --wm_model exp --attack paraphrase --task translation --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack_exp.py --wm_model exp --attack frequency --task translation --max_examples 100

CUDA_VISIBLE_DEVICES=2,3 uv run attack_exp.py --wm_model exp --attack none --task qa --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack_exp.py --wm_model exp --attack detection --task qa --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack_exp.py --wm_model exp --attack paraphrase --task qa --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack_exp.py --wm_model exp --attack frequency --task qa --max_examples 100

CUDA_VISIBLE_DEVICES=2,3 uv run attack_exp.py --wm_model exp --attack none --task summarization --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack_exp.py --wm_model exp --attack detection --task summarization --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack_exp.py --wm_model exp --attack paraphrase --task summarization --max_examples 100
CUDA_VISIBLE_DEVICES=2,3 uv run attack_exp.py --wm_model exp --attack frequency --task summarization --max_examples 100