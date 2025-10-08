for model in InternLM2-Math-Plus-7B Deepseek-Math-RL-7B InternLM2-Math-Plus-1.8B; do
    for method in PPL SC RPC; do
        python main.py --dataset MATH --model $model --method $method --K 64
    done
    for dataset in MathOdyssey AIME OlympiadBench; do
        for method in PPL SC RPC; do
            python main.py --dataset $dataset --model $model --method $method --K 128
        done
    done
done