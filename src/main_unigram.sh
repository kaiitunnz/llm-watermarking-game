#!/bin/bash
# attack -> ['none', 'detection', 'paraphrase', 'frequency']
# task -> ['qa', 'translation', 'summarization']

CUDA_VISIBLE_DEVICES=0,1 uv run attack_unigram.py --wm_model unigram --attack none --task translation --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_unigram.py --wm_model unigram --attack detection --task translation --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_unigram.py --wm_model unigram --attack paraphrase --task translation --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_unigram.py --wm_model unigram --attack frequency --task translation --max_examples 100
                                       
CUDA_VISIBLE_DEVICES=0,1 uv run attack_unigram.py --wm_model unigram --attack none --task qa --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_unigram.py --wm_model unigram --attack detection --task qa --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_unigram.py --wm_model unigram --attack paraphrase --task qa --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_unigram.py --wm_model unigram --attack frequency --task qa --max_examples 100
                                       
CUDA_VISIBLE_DEVICES=0,1 uv run attack_unigram.py --wm_model unigram --attack none --task summarization --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_unigram.py --wm_model unigram --attack detection --task summarization --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_unigram.py --wm_model unigram --attack paraphrase --task summarization --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_unigram.py --wm_model unigram --attack frequency --task summarization --max_examples 100