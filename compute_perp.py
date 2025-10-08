import json
import metrics
import argparse
import numpy as np
import multiprocessing
from tqdm import trange
import signal, functools
import re, os, sys, random, time
from fraction import Fraction
from data_processing.answer_extraction import *
from functools import lru_cache
from eval.eval_script import *
MAX_INT = sys.maxsize
INVALID_ANS = "[Invalid]"
INF = 1e9

__all__ = [
    "check_equal",
    "check_equal_without_timeout",
    "numberic_compare",
    "Evaluator",
]

def timeout(sec):
    """
    timeout decorator
    :param sec: function raise TimeoutError after ? seconds
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):

            def _handle_timeout(signum, frame):
                err_msg = f"Function {func.__name__} timed out after {sec} seconds"
                raise TimeoutError(err_msg)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(sec)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapped_func

    return decorator


@timeout(1)
def check_equal_without_timeout(ans_1, ans_2):
    return math_equal(ans_1, ans_2)

def check_equal(ans_1, ans_2, cache_dict=None):
    try:
        if cache_dict is not None:
            key = str(ans_1) + "<##>" + str(ans_2)
            if key in cache_dict: return cache_dict[key]
            print("Miss")
        return check_equal_without_timeout(ans_1, ans_2)
    except TimeoutError as e:
        return False

def numberic_compare(ai, aj, ci, cj, cache_dict=None):
    return check_equal(ai, aj, cache_dict)

def prep_evaluator(
    predicts, completions, perplexities, answer, equal_func, check_equal
):
    m = len(predicts)

    # Compute maximum probability
    max_perplexity = -INF
    max_perplexity_count = 0.0
    for i in range(m):
        if perplexities[i] > max_perplexity:
            max_perplexity = perplexities[i]
            max_perplexity_count = 0.0
        if perplexities[i] >= max_perplexity:
            max_perplexity_count += 1.0

    # Compute accuracy
    correct, answers = 0, []
    for i in range(m):
        ans_i = predicts[i]
        answers.append([ans_i, np.exp(perplexities[i]), check_equal(ans_i, answer)])
        if perplexities[i] < max_perplexity: continue
        if check_equal(ans_i, answer):
            correct += 1.0 / max_perplexity_count

    return correct, answers

class Evaluator:
    def __init__(self):
        self.name = "Perplexity"

    def process(self, json_file, cache_file, equal_func, evaluator, K, seed=0):
        # with open(file_path, 'r', encoding='utf-8') as f:
        #     results = json.load(f)
        results = json_file
        n = len(results["predict"])
        m = len(results["predict"][0])
        indices = list(range(m))
        random.seed(seed)
        random.shuffle(indices)
        indices = indices[: K]

        if cache_file is not None:
            def cache_equal_func(ai, aj, ci, cj):
                return equal_func(ai, aj, ci, cj, cache_file)
            def cache_check_equal(ai, aj):
                return check_equal(ai, aj, cache_file)
        else:
            cache_equal_func = equal_func
            cache_check_equal = check_equal


        predicts, completions, perplexities, answers = [], [], [], []
        for i in range(0, n):
            predicts.append([results["predict"][i][j] for j in indices])
            completions.append([results["completion"][i][j] for j in indices])
            perplexities.append([results["mean_logprob"][i][j] for j in indices])
            answers.append(results["answer"][i])
        n = len(predicts)

        start_time = time.time()
        outputs = []
        for idx in trange(n):
            res = evaluator(
                predicts[idx],
                completions[idx],
                perplexities[idx],
                answers[idx],
                cache_equal_func,
                cache_check_equal,
            )
            outputs.append(res)
        print(f"Running Time with Single Process Mode with Seed #{seed}: {time.time() - start_time:.2f}S")

        for i in trange(n):
            m = len(outputs[i][1])
            for j in range(m):
                ans, prob, flag = outputs[i][1][j]
        maximum, max_bins = metrics.compute_maximum_metrics([x[1] for x in outputs])
        average, avg_bins = metrics.compute_average_metrics([x[1] for x in outputs])
        accs = np.mean([x[0] for x in outputs])
        return accs * 100.0, maximum, average, max_bins, avg_bins

    def worker(self, args):
        json_file, cache_file, K, seed = args
        acc, maximum, average, max_bins, avg_bins = self.process(
            json_file=json_file, 
            cache_file=cache_file,
            equal_func=numberic_compare, 
            evaluator=prep_evaluator,
            K=K, 
            seed=seed
        )
        return acc, maximum, average

    def solve(self, json_file, cache_file=None, repeats=10, K=128):
        accs, maxs, avgs = [], [], []
        with multiprocessing.Pool() as pool:
            results = pool.map(self.worker, [(json_file, cache_file, K, seed) for seed in range(repeats)])
        accs, maxs, _ = zip(*results)
        accs, maxs = np.array(accs), np.array(maxs)
        return {
            "Accuracy": f"{accs.mean():.2f} ± {accs.std():.2f}",
            "ECE": f"{maxs[:, 0].mean() * 100.0:.2f} ± {maxs[:, 0].std() * 100.0:.2f}",
        }
