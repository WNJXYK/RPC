import os
import argparse
from huggingface_hub import hf_hub_download
import json
from build_cache import cache
from compute_perp import Evaluator as PPLEvaluator
from compute_sc import SCEvaluator
from compute_rpc import RPCEvaluator

REPOID = {
    "MATH": "WNJXYK/MATH-Reasoning-Paths",
    "MathOdyssey": "WNJXYK/MathOdyssey-Reasoning-Paths",
    "AIME": "WNJXYK/AIME_1983_2024-Reasoning-Paths",
    "OlympiadBench": "WNJXYK/OlympiadBench-Reasoning-Paths"
}

EVALUATOR_MAP = {
    "PPL": PPLEvaluator,
    "SC": SCEvaluator,
    "RPC": RPCEvaluator
}

args = argparse.ArgumentParser()
args.add_argument("--dataset", type=str, choices=["MATH", "MathOdyssey", "AIME", "OlympiadBench"], default="MathOdyssey")
args.add_argument("--model", type=str, choices=["Deepseek-Math-RL-7B", "InternLM2-Math-Plus-1.8B", "InternLM2-Math-Plus-7B"], default="InternLM2-Math-Plus-7B")
args.add_argument("--K", type=int, default=128)
args.add_argument("--method", type=str, default="PPL", choices=["PPL", "SC", "RPC"])
args = args.parse_args()

repo_id = REPOID[args.dataset]
filename = args.model + ".json"

# Download sampled reasoning paths from Hugging Face
try:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with open(file_path, 'r', encoding='utf-8') as f:
        json_file = json.load(f)
    print(f"Load sampled reasoning paths {filename} from {repo_id} successfully!")
except Exception as e:
    print(f"Failed to load sampled reasoning paths {filename} from {repo_id}: {e}")

# Build cache for checking equality
cache_path = file_path.replace(".json", ".cache.json")
cache(json_file, cache_path)
with open(cache_path, 'r', encoding='utf-8') as f:
    cache_file = json.load(f)

# Run!
results = EVALUATOR_MAP[args.method]().solve(json_file=json_file, cache_file=cache_file, K=args.K)

# Report results
result_str = f"{args.method} {args.dataset} {args.model} {args.K} {results}"
with open("results.txt", "a") as f:
    f.write(result_str + "\n")
print(result_str)