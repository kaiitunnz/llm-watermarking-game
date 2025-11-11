"""
End-to-end tests for LLM watermarking generation.

Tests the complete workflow: model loading, watermarked generation, and detection.
"""
import time
import pytest
from tasks import load_dataset
import evaluate
from tasks import *
from wmgame.watermark.base import DetectionConfig
from wmgame.watermark.kgw import KGWWatermarkedLLM, KGWDetector
from wmgame.watermark.unigram import UnigramWatermarkedLLM, UnigramDetector
from wmgame.watermark.exp import ExpWatermarkedLLM, ExpDetector
import argparse
import logging
import os


# Use a small model for testing
MODEL_NAME = "meta-llama/Llama-3.1-8B"

def main():
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str, default='none', 
                       choices=['none', 'detection', 'paraphrase', 'frequency'],
                       help='Attack method to use')
    parser.add_argument('--task', type=str, default='qa',
                       choices=['qa', 'translation', 'summarization'],
                       help='Task to evaluate on')
    parser.add_argument('--max_examples', type=int, default=50,
                       help='Maximum number of examples to process')
    parser.add_argument('--wm_model', type=str, default='kgw',
                       choices=['kgw', 'unigram', 'exp'],)
    args = parser.parse_args()

    attack_method = args.attack
    task = args.task
    max_examples = args.max_examples
    wm_model = args.wm_model

    # Set up logging
    log_dir = "outputs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"attack_results_{args.wm_model}_{args.attack}_{args.task}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.FileHandler(log_filename)  # Output to file
        ]
    )
    logger = logging.getLogger(__name__)

    start_time = time.time()
    
    # Initialize metrics
    comet = evaluate.load("comet")
    bertscore = evaluate.load("bertscore")
    total_examples = 0
    successful_attacks = 0
    total_comet = 0.0
    total_bert_f1 = 0.0
    total_generation_time = 0.0

    # Load examples based on task
    if task == "qa":
        all_prompts = load_opengen_qa_from_wikitext(max_examples=max_examples)
    elif task == "translation":
        all_prompts = load_translation_examples(max_examples=max_examples)
    elif task == "summarization":
        all_prompts = load_summarization_examples(max_examples=max_examples)
    else:
        raise ValueError(f"Unknown task: {task}")

    # Initialize LLM and detector
    if wm_model == "kgw":
        llm = KGWWatermarkedLLM(MODEL_NAME)
        detector = KGWDetector(tokenizer=llm.tokenizer, gamma=0.5, z_threshold=2.0, config=DetectionConfig(threshold=2.0))
    elif wm_model == "unigram":
        llm = UnigramWatermarkedLLM(MODEL_NAME)
        detector = UnigramDetector(tokenizer=llm.tokenizer, fraction=0.5, vocab_size=llm.tokenizer.vocab_size, watermark_key=0, config=DetectionConfig(threshold=2.0))
    elif wm_model == "exp":
        llm = ExpWatermarkedLLM(MODEL_NAME)
        detector = ExpDetector(tokenizer=llm.tokenizer, n=256, k=1, gamma=0.5, seed=0, vocab_size=llm.tokenizer.vocab_size, config=DetectionConfig(threshold=0.05))
    else:
        raise ValueError(f"Unknown watermark model: {wm_model}")

    for i, (prompt, target) in enumerate(all_prompts, 1):
        example_start_time = time.time()

        if attack_method == "detection":
            output = llm.generate_with_detection_attack(
                prompt=prompt,
                detector=detector,
                gamma=0.3,
                delta=4.0,
                max_new_tokens=128,
                do_sample=False,
            )
        elif attack_method == "frequency":
            output = llm.generate_with_frequency_attack(
                prompt=prompt,
                num_samples=10,
                reduction_factor=0.1,
                gamma=0.3,
                delta=4.0,
                max_new_tokens=128,
                do_sample=False,
            )
        elif attack_method == "paraphrase":
            output = llm.generate_with_paraphrase_attack(
                prompt=prompt,
                gamma=0.3,
                delta=4.0,
                max_new_tokens=128,
                do_sample=False,
            )
        elif attack_method == "none":
            output = llm.generate_with_watermark(
                prompt=prompt,
                gamma=0.3,
                delta=4.0,
                max_new_tokens=128,
                do_sample=False,
            )
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")

        generated_text = llm.tokenizer.decode(output[0], skip_special_tokens=True)
        
        if task=="translation":
            comet_output = comet.compute(
                predictions=[generated_text],
                references=[target],
                sources=[prompt]  # COMET needs source text for translation
            )
            comet_score = comet_output["scores"][0]
            total_comet += comet_score
        else:
            # Calculate BERTScore F1
            bert_res = bertscore.compute(
                predictions=[generated_text],
                references=[target],
                model_type="microsoft/deberta-xlarge-mnli",  # or "bert-base-uncased" 
            )
            bert_f1 = float(bert_res["f1"][0])
            total_bert_f1 += bert_f1
        
        logger.info("\n" + "=" * 60)
        logger.info(f"Target: {target}")
        logger.info(f"Generated text: {generated_text}")

        # Detect watermark
        result = detector.detect(generated_text)
        logger.info(f"Detection result: z={result.z_score:.3f}, passed={result.passed}")
        if task == "translation":
            logger.info(f"COMET score: {comet_score:.3f}")
        else:
            logger.info(f"BERTScore F1: {bert_f1:.3f}")
        
        logger.info("\n" + "=" * 60)

        # Track attack success
        total_examples += 1
        if not result.passed:  
            successful_attacks += 1
            
        #logger.info("-" * 60)
        example_time = time.time() - example_start_time
        total_generation_time += example_time
        logger.info(f"One step executing time: {example_time:.3f} s")

    total_time = time.time() - start_time

    # Print summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("ATTACK SUMMARY")
    logger.info("-" * 60)
    logger.info(f"Attack method: {attack_method}")
    logger.info(f"Total examples: {total_examples}")
    logger.info(f"Attack success rate: {(successful_attacks/total_examples)*100:.2f}%")
    if task == "translation":
        logger.info(f"Average COMET score: {(total_comet/total_examples):.3f}")
    else:
        logger.info(f"Average BERTScore F1: {(total_bert_f1/total_examples):.3f}")
    logger.info(f"Total execution time: {total_time:.2f}s")
    logger.info("=" * 60)

#class TestKGWGeneration:
    #"""Test KGW watermarking end-to-end."""
    
    #def test_kgw_generate_with_watermark_method(self):
        #"""Test KGW direct generation method."""
        #llm = KGWWatermarkedLLM(MODEL_NAME)
        #test_prompt = "The quick brown fox"
        #attack_method = "frequency" # "frequency" or "detection", or "paraphrase"

        #detector = KGWDetector(
            #tokenizer=llm.tokenizer,
            #gamma=0.5,
            #config=DetectionConfig(threshold=2.0),
        #)
        
        ## Use generate_with_watermark method
        #if attack_method == "detection":
             #output = llm.generate_with_detection_attack(
                 #prompt=test_prompt,
                 #detector=detector,
                 #gamma=0.5,
                 #delta=2.0,
                 #max_new_tokens=20,
                 #do_sample=False,
             #)
        #elif attack_method == "frequency":
            ## TODO: This is spoofing attack so need to modify to conduct watermark removal
             #output = llm.generate_with_frequency_attack(
                 #prompt=test_prompt,
                 #num_samples=10,
                 #gamma=0.5,
                 #delta=2.0,
                 #max_new_tokens=20,
                 #do_sample=False,
             #)
        #elif attack_method == "paraphrase":
            #output = llm.generate_with_paraphrase_attack(
                #prompt=test_prompt,
                #gamma=0.5,
                #delta=2.0,
                #max_new_tokens=20,
                #do_sample=False,
            #)
        #else:
            #raise ValueError(f"Unknown attack method: {attack_method}")

        #generated_text = llm.tokenizer.decode(output[0], skip_special_tokens=True)
        
        #print(f"\n✓ Generated text: {generated_text}")
        
        ## Verify generation happened
        #assert len(generated_text) > len(test_prompt)
        
        
        #result = detector.detect(generated_text)
        
        #print(f"  Detection - z_score: {result.z_score:.3f}, passed: {result.passed}")
        
        #assert result.z_score is not None



class TestDetectorAPIs:
    """Test detector APIs work correctly."""
    
    def test_kgw_detector_basic(self):
        """Test KGW detector basic functionality."""
        llm = KGWWatermarkedLLM(MODEL_NAME)
        
        detector = KGWDetector(
            tokenizer=llm.tokenizer,
            gamma=0.5,
            config=DetectionConfig(threshold=2.0),
        )
        
        # Test with some text
        text = "This is a test sentence with more words to have enough tokens for detection."
        result = detector.detect(text)
        
        assert result is not None
        assert hasattr(result, 'z_score')
        assert hasattr(result, 'passed')
        
        print(f"\n✓ KGW detector works: z_score={result.z_score}")
        
    def test_unigram_detector_basic(self):
        """Test Unigram detector basic functionality."""
        llm = UnigramWatermarkedLLM(MODEL_NAME)
        
        detector = UnigramDetector(
            tokenizer=llm.tokenizer,
            fraction=0.5,
            vocab_size=llm.tokenizer.vocab_size,
            watermark_key=0,
            config=DetectionConfig(threshold=2.0),
        )
        
        # Test with some text
        text = "This is a test sentence with more words to have enough tokens for detection."
        result = detector.detect(text)
        
        assert result is not None
        assert hasattr(result, 'z_score')
        assert hasattr(result, 'green_fraction')
        assert hasattr(result, 'passed')
        
        print(f"\n✓ Unigram detector works: z_score={result.z_score}, green_frac={result.green_fraction}")
        
    def test_exp_detector_basic(self):
        """Test EXP detector basic functionality."""
        llm = ExpWatermarkedLLM(MODEL_NAME)
        
        detector = ExpDetector(
            tokenizer=llm.tokenizer,
            n=256,
            k=1,
            gamma=0.5,
            seed=0,
            vocab_size=llm.tokenizer.vocab_size,
            config=DetectionConfig(threshold=0.05),
        )
        
        # Test with some text
        text = "This is a test sentence with more words to have enough tokens for detection."
        result = detector.detect(text)
        
        assert result is not None
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'passed')
        
        print(f"\n✓ EXP detector works: p_value={result.p_value}")


#if __name__ == "__main__":
    #pytest.main([__file__, "-v", "-s"])

if __name__ == "__main__":
    main()

