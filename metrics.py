import numpy as np

EPS = 1e-10
NLLEPS = 1e-6

def compute_maximum_metrics(predicts, n_bins=10):
    n = len(predicts)
    acc, cnf, siz = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    brier_score = []
    negative_ll = []

    for idx in range(n):
        m = len(predicts[idx])

        # Compute maximum probabilities and corresponding counts within each problem
        max_prob, max_prob_counts = -1e6, 0
        for i in range(m):
            ans, prob, flag = predicts[idx][i]
            if prob > max_prob:
                max_prob, max_prob_counts = prob, 0
            if prob >= max_prob - EPS:
                max_prob_counts += 1
        # print(max_prob, max_prob_counts)
        # Compute the maximum accuracy for each problem as well as the ECE metric
        vote_acc = 0
        for i in range(m):
            ans, prob, flag = predicts[idx][i]
            if prob < max_prob:
                continue
            if np.isnan(prob):
                continue
            if flag:
                vote_acc += 1.0 / max_prob_counts
            # Compute Expected Calibration Error
            for cur in range(n_bins):
                lower, upper = cur / n_bins, (cur + 1) / n_bins
                if lower < max_prob <= upper:
                    if flag:
                        acc[cur] += 1.0 / max_prob_counts
                    cnf[cur] += prob / max_prob_counts
                    siz[cur] += 1.0 / max_prob_counts

        # Compute Brier Score
        brier_score.append((vote_acc - max_prob) ** 2)

        # Compute Negative Likelihhod
        cliped_max_prob = max(min(max_prob, 1 - NLLEPS), NLLEPS)
        cliped_vote_acc = max(min(vote_acc, 1 - NLLEPS), NLLEPS)
        negative_ll.append(
            -np.log(cliped_max_prob) * cliped_vote_acc
            - np.log(1 - cliped_max_prob) * (1 - cliped_vote_acc)
        )

    # Turn each metrics into values
    ece = 0
    for cur in range(n_bins):
        if siz[cur] > 0:
            acc[cur] = acc[cur] / siz[cur]
            cnf[cur] = cnf[cur] / siz[cur]
        ece += siz[cur] * np.abs(acc[cur] - cnf[cur])
        # print(siz[cur], acc[cur], cnf[cur])
    ece = ece / sum(siz)
    bs = np.mean(brier_score)
    nll = np.mean(negative_ll)

    return (ece, bs, nll), (acc, cnf, siz)


def compute_average_metrics(predicts, n_bins=10):
    n = len(predicts)
    acc, cnf, siz = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    brier_score = []
    negative_ll = []

    for idx in range(n):
        m = len(predicts[idx])

        problem_brier_score = []
        problem_negative_ll = []
        for i in range(m):
            ans, prob, flag = predicts[idx][i]
            # Compute Expected Calibration Error
            for cur in range(n_bins):
                lower, upper = cur / n_bins, (cur + 1) / n_bins
                if lower < prob <= upper:
                    if flag:
                        acc[cur] += 1.0 / m
                    cnf[cur] += prob / m
                    siz[cur] += 1.0 / m

            # Compute Brier Score
            problem_brier_score.append(((1 if flag else 0) - prob) ** 2)

            # Compute Negative Likelyhood
            cliped_max_prob = max(min(prob, 1 - NLLEPS), NLLEPS)
            cliped_vote_acc = max(min(1 if flag else 0, 1 - NLLEPS), NLLEPS)
            problem_negative_ll.append(
                -np.log(cliped_max_prob) * cliped_vote_acc
                - np.log(1 - cliped_max_prob) * (1 - cliped_vote_acc)
            )

        brier_score.append(np.mean(problem_brier_score))
        negative_ll.append(np.mean(problem_negative_ll))

    ece = 0
    for cur in range(n_bins):
        if siz[cur] > 0:
            acc[cur] = acc[cur] / siz[cur]
            cnf[cur] = cnf[cur] / siz[cur]
        ece += siz[cur] * np.abs(acc[cur] - cnf[cur])
    ece = ece / sum(siz)
    bs = np.mean(brier_score)
    nll = np.mean(negative_ll)

    return (ece, bs, nll), (acc, cnf, siz)
