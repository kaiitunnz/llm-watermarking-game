#!/bin/bash
# attack -> ['none', 'detection', 'paraphrase', 'frequency']
# task -> ['qa', 'translation', 'summarization']

CUDA_VISIBLE_DEVICES=0,1 uv run attack_kgw.py --wm_model kgw --attack none --task translation --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_kgw.py --wm_model kgw --attack detection --task translation --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_kgw.py --wm_model kgw --attack paraphrase --task translation --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_kgw.py --wm_model kgw --attack frequency --task translation --max_examples 100
                            
CUDA_VISIBLE_DEVICES=0,1 uv run attack_kgw.py --wm_model kgw --attack none --task qa --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_kgw.py --wm_model kgw --attack detection --task qa --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_kgw.py --wm_model kgw --attack paraphrase --task qa --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_kgw.py --wm_model kgw --attack frequency --task qa --max_examples 100
                            
CUDA_VISIBLE_DEVICES=0,1 uv run attack_kgw.py --wm_model kgw --attack none --task summarization --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_kgw.py --wm_model kgw --attack detection --task summarization --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_kgw.py --wm_model kgw --attack paraphrase --task summarization --max_examples 100
CUDA_VISIBLE_DEVICES=0,1 uv run attack_kgw.py --wm_model kgw --attack frequency --task summarization --max_examples 100