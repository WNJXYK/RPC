
# [NeurIPS 2025] A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning

Official Repository for NeurIPS 2025 Paper: "A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning"

<div align="center">
<a href="https://arxiv.org/pdf/2502.00511">📄 [Paper]</a>
&nbsp;
<a href="https://wnjxyk.github.io/RPC">🌐 [Project]</a>
&nbsp;
<a href="https://huggingface.co/collections/WNJXYK/mathematical-llm-reasoning-paths-68e4c4e32e3ad7fa0fcad77a">🤗 [Data Collection]</a>
</div>


## 🛠️ 1. Environment Setup

We provide two ways to create the Python environment for this repository. Please choose one of the following methods:

### 1.1. Using Python virtual environment:

```bash
python -m venv rpc
source rpc/bin/activate
pip install -r requirements.txt 
```

### 1.2. Using Conda environment:

```bash
conda create -n rpc python=3.9
conda activate rpc
pip install -r requirements.txt
```

## 🚀 2. Reproducing Experiments

### 2.1. Single Experiment

Run evaluation with specific parameters:

```bash
python main.py --dataset MathOdyssey --model InternLM2-Math-Plus-7B --method RPC --K 128
```

**Parameters:**
- `--dataset`: Choose from `MATH`, `MathOdyssey`, `AIME`, `OlympiadBench`
- `--model`: Choose from `Deepseek-Math-RL-7B`, `InternLM2-Math-Plus-1.8B`, `InternLM2-Math-Plus-7B`
- `--method`: Choose from `PPL` (Perplexity), `SC` (Self-Consistency), `RPC` (our method)
- `--K`: Number of reasoning paths to sample (`128` for `MathOdyssey`, `AIME`, `OlympiadBench`, and `64` for `MATH`)

### 2.2. Batch Experiments

Run comprehensive evaluation across multiple settings:

```bash
bash all_exps.sh
```

This will evaluate all method-dataset-model combinations and save results to `results.txt`.

### 2.3. Hints

1. If you cannot download data from Hugging Face directly, please use [Hugging Mirror](https://hf-mirror.com/) instead.
2. It may take some time to generate the cache for checking answer equality when running each dataset for the first time.

## 📚 3. BibTex

```bibtex
@inproceedings{zhou24theoretical,
      author    = {Zhou, Zhi and Tan, Yuhao and Li, Zenan and Yao, Yuan and Guo, Lan-Zhe and Li, Yu-Feng and Ma, Xiaoxing},
      title     = {A Theorecial Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning},
      booktitle = {Advances in Neural Information Processing Systems},
      year      = {2025},
    }
```