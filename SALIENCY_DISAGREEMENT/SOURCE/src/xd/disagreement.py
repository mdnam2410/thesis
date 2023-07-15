import numpy as np
import torch


def compute_mean_disagreement(exp1, exp2, metric, **options):
    metrics = []
    def has_nan(x): return isinstance(x, np.ndarray) and np.isnan(x).any()
    if has_nan(exp1) or has_nan(exp2):
        return float('nan')
    for e1, e2 in zip(exp1, exp2):
        metrics.append(metric(e1, e2, **options))
    return np.array(metrics).mean()


def compute_heatmap(methods, explanations, metric, **options):
    """Compute disagreement heatmap for one blackbox
    """
    size = len(methods)
    result = dict()
    result['data'] = np.empty((size, size))
    result['methods'] = methods

    for j in range(size):
        for i in range(size):
            exp1 = explanations[methods[i]]
            exp2 = explanations[methods[j]]

            if isinstance(exp1, torch.Tensor):
                exp1 = exp1.cpu().numpy()

            if isinstance(exp2, torch.Tensor):
                exp2 = exp2.cpu().numpy()

            if i >= j:
                result['data'][i, j] = compute_mean_disagreement(
                    exp1, exp2, metric, **options)
            else:
                result['data'][i, j] = result['data'][j, i]
    return result


def compute_heatmaps(explanations, metric, **options):
    result = dict()
    for k, v in explanations.items():
        methods = list(v.keys())
        methods.sort()
        result[k] = compute_heatmap(methods, v, metric, **options)
    return result
