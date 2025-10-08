from compute_perp import check_equal
import multiprocessing, json, os, time

def solve(predict, answer):
    cache_dict = {}
    m = len(predict)

    for i in range(m):
        key = str(predict[i]) + "<##>" + str(answer)
        rev_key = str(answer) + "<##>" + str(predict[i])
        if key in cache_dict or rev_key in cache_dict:
            continue
        val = check_equal(predict[i], answer)
        cache_dict[key] = val
        cache_dict[rev_key] = val
    
    for i in range(m):
        for j in range(m):
            key = str(predict[i]) + "<##>" + str(predict[j])
            rev_key = str(predict[j]) + "<##>" + str(predict[i])
            if key in cache_dict or rev_key in cache_dict:
                continue
            val = check_equal(predict[i], predict[j])
            cache_dict[key] = val
            cache_dict[rev_key] = val

    return cache_dict

def cache(data, cache_path):
    if os.path.exists(cache_path): 
        print(f"Cache file {cache_path} exists, skip!")
        return
    start_time = time.time()
    predicts = data["predict"]
    answers = data["answer"]
    n = len(predicts)
    cache_dict = {}
    with multiprocessing.Pool() as pool:
        results = pool.starmap(
            solve, [(predicts[i], answers[i]) for i in range(n)]
        )
    for result in results:
        cache_dict.update(result)
    with open(cache_path, "w") as fw:
        json.dump(cache_dict, fw)
    print(f"Cache file {cache_path} built in {time.time() - start_time:.2f}S")