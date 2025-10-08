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
from eval.eval_script import *
from compute_perp import Evaluator, numberic_compare
MAX_INT = sys.maxsize
INVALID_ANS = "[Invalid]"

__all__ = ["DSU"]

class DSU:
    def __init__(self, n):
        self.n = n
        self.father = [i for i in range(n)]
        self.size = [1 for i in range(n)]
        self.attr = [{} for i in range(n)]

    def get_father(self, x):
        if self.father[x] == x:
            return x
        self.father[x] = self.get_father(self.father[x])
        return self.father[x]

    def merge(self, x, y):
        fx = self.get_father(x)
        fy = self.get_father(y)
        if fx == fy:
            return
        self.father[fy] = fx
        self.size[fx] += self.size[fy]
        self.size[fy] = 0
        for key in self.attr[fy].keys():
            if key not in self.attr[fx]:
                self.attr[fx][key] = self.attr[fy][key]
            else:
                self.attr[fx][key] |= self.attr[fy][key]
        self.attr[fy] = {}


def sc_evaluator(predicts, completions, perplexities, answer, equal_func, check_equal):
    m = len(predicts)
    dsu = DSU(m)

    # Merge answer for self-consistency
    for i in range(m):
        if dsu.get_father(i) != i:
            continue
        for j in range(i):
            ans_i = predicts[i]
            ans_j = predicts[j]
            completion_i = completions[i]
            completion_j = completions[j]
            if equal_func(ans_i, ans_j, completion_i, completion_j):
                dsu.merge(i, j)

    # Compute majority votes
    max_size, max_size_count = 0, 0
    for i in range(m):
        if dsu.get_father(i) != i:
            continue
        if dsu.size[i] > max_size:
            max_size = dsu.size[i]
            max_size_count = 0
        if dsu.size[i] == max_size:
            max_size_count += 1

    # Compute accuracy
    correct, answers = 0, []
    for i in range(m):
        if dsu.get_father(i) != i:
            continue
        ans_i = predicts[i]
        answers.append([ans_i, dsu.size[i] / m, check_equal(ans_i, answer)])
        if dsu.size[i] < max_size:
            continue
        if check_equal(ans_i, answer):
            correct += 1.0 / max_size_count

    # Normalize probabilities
    sum_proba = np.sum([x[1] for x in answers])
    for i in range(len(answers)):
        answers[i][1] /= sum_proba

    return correct, answers


class SCEvaluator(Evaluator):
    def __init__(self):
        self.name = "Self-Consistency"

    def worker(self, args):
        json_file, cache_file, K, seed = args
        acc, maximum, average, max_bins, avg_bins = self.process(
            json_file=json_file, 
            cache_file=cache_file,
            equal_func=numberic_compare, 
            evaluator=sc_evaluator,
            K=K, 
            seed=seed
        )
        return acc, maximum, average
