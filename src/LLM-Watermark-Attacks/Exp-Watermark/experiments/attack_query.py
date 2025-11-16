import sys
sys.path.append('./')

import tqdm, json, torch, pickle, argparse, os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from watermarking.detection import phi, fast_permutation_test_query
from watermarking.gumbel.score import gumbel_edit_score
from watermarking.gumbel.sampler import gumbel_query
from watermarking.gumbel.key import gumbel_key_func

def read_file(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f.read().strip().split("\n")]


def detect(sentence, seed, tokenizer, args, test):
    if isinstance(sentence, str):
        tokens = tokenizer.encode(sentence, return_tensors='pt', truncation=True, max_length=2048)[0]
    else:
        tokens = sentence
    token_len = tokens.shape[-1]
    if token_len > args.m:
        token_len = args.m
    null_id = (token_len - 1) // 5
    cur_k = (null_id + 1) * 5
    if tokens.shape[-1] < cur_k:
        tokens = torch.nn.functional.pad(tokens,(cur_k-tokens.shape[-1],0),"constant",0)
    else:
        tokens = tokens[:args.m]

    pval = test(tokens, seed, cur_k, null_id)

    return pval

def query_sampling(probs, prev, tokenizer, seed, args, test):
    sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
    sorted_probs = sorted_probs.float()
    cumulative_sum = torch.cumsum(sorted_probs, dim=0)
    cutoff_idx = torch.where(cumulative_sum >= 0.99)[0][0]
    if args.action == 'removal':
        cutoff_idx = min([cutoff_idx, 4])
    elif args.action == 'spoofing' or args.action == 'spoofing-defense':
        cutoff_idx = 2
    tokens_list = sorted_indices[:cutoff_idx+1]
    p_vals = []

    query_flag = False

    if prev is not None:
        prev = prev[0].detach().cpu()
        if args.action == 'removal':
            if prev.shape[0] > 0:
                query_flag = True
        elif args.action == 'spoofing' or args.action == 'spoofing-defense':
            if prev.shape[0] > 1:
                query_flag = True
    if len(tokens_list) == 1 and query_flag:
        total_query = -1
    else:
        total_query = 0
    if query_flag:
        for idx in range(len(tokens_list)):
            if sorted_probs[idx] < 0.15 and idx != 0 and args.action == 'removal' and p_vals[0] > 0.1:
                break
            token = tokens_list[idx]
            token = token.reshape((1))
            if prev is not None:
                new_list = torch.cat((prev, token))
            else:
                new_list = token
            p_vals.append(detect(new_list, seed, tokenizer, args, test))
            total_query += 1

        if args.action == 'spoofing' or args.action == 'spoofing-defense':
            max_z_score = min(p_vals)
        elif args.action == 'removal':
            max_z_score = max(p_vals)

        selected_word_index = p_vals.index(max_z_score)
        selected_word = tokens_list[selected_word_index]
    else:
        selected_word = tokens_list[0]
    return selected_word.reshape((1, 1)), total_query


def generate_wm(model, prompt, vocab_size, tokenizer, seed, args, test):
    inputs = prompt.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    outputs = None
    total_query = 0
    num_tokens = 0
    if args.action == 'removal':
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        xi, pi = gumbel_key_func(generator, args.n, vocab_size)
        xi = xi.unsqueeze(0)
        pi = pi.unsqueeze(0)
    while True:
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)
        probs = torch.nn.functional.softmax(output.logits[:,-1, :vocab_size], dim=-1).cpu()
        if args.action == 'removal':
            probs = gumbel_query(probs, pi, xi[:, (num_tokens) % args.n])
        token, cur_query = query_sampling(probs, prev=outputs, tokenizer=tokenizer, seed=seed, args=args, test=test)
        total_query += cur_query
        token = token.to(model.device)
        inputs = torch.cat([inputs, token], dim=-1)
        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        if outputs is None:
            outputs = token
        else:
            outputs = torch.cat([outputs, token], dim=-1)
        num_tokens += 1
        if 'opt' in args.model_name_or_path:
            if num_tokens > args.max_new_tokens:
                break
        else:
            if token == model.config.eos_token_id or num_tokens > args.max_new_tokens:
                break

    return outputs.detach().cpu(), total_query, num_tokens


def main(args):
    if 'Llama' in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
    vocab_size = len(tokenizer)

    dist = lambda x,y : gumbel_edit_score(x,y,gamma=args.gamma)

    test_stat = lambda tokens,n,k,generator,vocab_size,null=False : phi(tokens=tokens,
                                                                        n=n,
                                                                        k=k,
                                                                        generator=generator,
                                                                        key_func=gumbel_key_func,
                                                                        vocab_size=vocab_size,
                                                                        dist=dist,
                                                                        null=null,
                                                                        normalize=False,)
    
    null_results_list = []
    for i in range(14):
        cur_id = (i + 1) * 5
        file_name = './results/null_tests_' + str(cur_id) + '.pkl'
        null_results = pickle.load(open(file_name, "rb"))
        null_results = torch.sort(torch.tensor(null_results)).values
        null_results_list.append(null_results.clone())
    test = lambda tokens,seed,k,null_id : fast_permutation_test_query(tokens,
                                                    vocab_size,
                                                    args.n,
                                                    k,
                                                    seed,
                                                    test_stat,
                                                    null_results_list,
                                                    null_id,
                                                    args)

    model_type = 'llama-7b' if 'llama' in args.model_name_or_path else 'opt' if 'opt' in args.model_name_or_path else None
    if not model_type:
        return

    base_path = f'./{model_type}-results'

    os.makedirs(base_path, exist_ok=True)

    if args.action == 'removal':
        os.makedirs(f'{base_path}/api_removal', exist_ok=True)
        data = read_file(args.data_file)

        if 'llama' in args.model_name_or_path:
            model = LlamaForCausalLM.from_pretrained(
                args.model_name_or_path,
                return_dict=True,
                load_in_8bit=False,
                device_map='auto',
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto', torch_dtype=torch.float16,)
        args.device = model.device

        watermark_removal_num_queries = []
        watermark_removal_results = []
        watermark_removal_num_tokens = []
        watermark_removal_attack_scores = []

        watermark_removal_num_queries_file_name = f'{base_path}/api_removal/watermark_removal_num_queries.pkl'
        watermark_removal_attack_scores_file_name = f'{base_path}/api_removal/watermark_removal_attack_scores.pkl'
        watermark_removal_results_file_name = f'{base_path}/api_removal/watermark_removal_results.pkl'
        watermark_removal_num_tokens_file_name = f'{base_path}/api_removal/watermark_removal_num_tokens.pkl'
        torch.manual_seed(args.wm_key)
        seeds = torch.randint(2**32, (args.T,))

        for idx in tqdm.tqdm(range(args.T)):
            cur_data = data[idx]
            if "gold_completion" not in cur_data and 'targets' not in cur_data:
                continue
            else:
                prefix = cur_data['prefix']
            text = prefix

            tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048)
            watermarked_tokens, total_query, num_tokens = generate_wm(model, tokens, len(tokenizer), tokenizer, seeds[idx], args, test)
            watermarked_text = tokenizer.decode(watermarked_tokens[0], skip_special_tokens=True)

            if len(watermarked_text) > 5:
                detect_z_score = detect(watermarked_text, seeds[idx], tokenizer, args, test)

                watermark_removal_num_queries.append(total_query)
                watermark_removal_attack_scores.append(detect_z_score)
                watermark_removal_results.append(watermarked_text)
                watermark_removal_num_tokens.append(num_tokens)

            if (idx + 1) % 10 == 0:
                with open(watermark_removal_num_queries_file_name, 'wb') as f:
                    pickle.dump(watermark_removal_num_queries, f)

                with open(watermark_removal_attack_scores_file_name, 'wb') as f:
                    pickle.dump(watermark_removal_attack_scores, f)

                with open(watermark_removal_results_file_name, 'wb') as f:
                    pickle.dump(watermark_removal_results, f)

                with open(watermark_removal_num_tokens_file_name, 'wb') as f:
                    pickle.dump(watermark_removal_num_tokens, f)
        print(watermark_removal_attack_scores)
        print(watermark_removal_results)
        print(watermark_removal_num_queries)
        print(watermark_removal_num_tokens)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--wm_key", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=70)

    parser.add_argument('--data_file', default='../data/OpenGen/inputs.jsonl', type=str, 
            help='a file containing the document to test')
    parser.add_argument('--action', default='spoofing', type=str,
            help='an optional prompt for generation')
    
    parser.add_argument("--epsilon", type=float, default=1.4)
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--sigma", type=float, default=4.0)

    parser.add_argument('--gamma',default=0.0,type=float)
    parser.add_argument('--load_nulltest',default="./results/null_tests.pkl",type=str)
    parser.add_argument('--m',default=70,type=int)
    parser.add_argument('--k',default=0,type=int)
    parser.add_argument('--n',default=256,type=int)
    parser.add_argument('--T',default=500,type=int)

    args = parser.parse_args()

    if args.k == 0: 
        args.k = args.m # k is the block size (= number of tokens)
    else:
        args.k = args.k
    args.max_new_tokens = args.m
    main(args)
