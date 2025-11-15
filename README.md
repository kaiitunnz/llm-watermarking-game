# Installation

1. Make sure you have `uv` installed. See the installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
2. Run the following command to install the dependencies.
   ```sh
   uv sync
   ```

# How to run

```sh
bash scripts/main_[Watermark_scheme].sh
```

## Mixed-strategy evaluation

To evaluate Nash or Stackelberg equilibria, prepare a JSON file that follows this schema:

```json
{
  "solution_type": "<nash|stackelberg>",
  "defender_strategy": {"<watermark-scheme>": prob, ...},
  "attacker_strategy": {"<attack-scheme>": prob, ...}
}
```

Then, run the evaluation script

```sh
bash scripts/eval.sh ./strategies/nash_sample.json
```

Example Nash strategy:

```json
{
  "solution_type": "nash",
  "defender_strategy": { "kgw": 0.5, "unigram": 0.25, "exp": 0.25 },
  "attacker_strategy": {
    "detection": 0.2,
    "paraphrase": 0.4,
    "translation": 0.4
  }
}
```

Example Stackelberg strategy:

```json
{
  "solution_type": "stackelberg",
  "defender_strategy": { "kgw": 0.5, "unigram": 0.25, "exp": 0.25 },
  "attacker_strategy": { "detection": 0, "paraphrase": 1, "translation": 0 }
}
```
