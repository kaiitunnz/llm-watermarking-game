# No Free Lunch in LLM Watermarking: Trade-offs in Watermarking Design Choices

----

Implementation of NeurIPS 24 paper. [*No Free Lunch in LLM Watermarking: Trade-offs in Watermarking Design Choices.*](https://openreview.net/pdf?id=rIOl7KbSkv])


----

Attacks on the [Exp](https://github.com/jthickstun/watermark) watermark.

----

### Watermark removal attacks exploiting the use of multiple watermark keys.

**Baseline detection results (no multiple keys)**

Run the following command to obtain watermark detection results without multiple keys. The results will be saved in `./opt-results/multiple_keys` folder.
For the LLAMA-2-7B model, pass `meta-llama/Llama-2-7b-hf` to the `--model_name_or_path` argument. Results for this model will be stored in the `./llama-7b-results/multiple_keys` folder.
```bash
python experiments/attack_multiple_keys.py --model_name_or_path facebook/opt-1.3b
```

**Attacks exploiting multiple keys**

To simulate watermark removal attacks using multiple keys, run the following command. You can adjust the `--num_keys` parameter to specify the number of keys used.
```bash
python experiments/attack_multiple_keys.py --model_name_or_path facebook/opt-1.3b --multiple_key --num_keys 7
```

----

### Attacks exploiting the public detection APIs

**Watermark removal attack exploiting the public detection API**

This attack attempts to remove watermarks using public detection APIs:
```bash
python experiments/attack_query.py --model_name_or_path facebook/opt-1.3b --action removal --data_file ...
```

Requires an input .jsonl file containing the input data. Format is {'prefix' : 'Input_sequence', 'gold_completion' : 'reference_output'}.

**Requirements**

