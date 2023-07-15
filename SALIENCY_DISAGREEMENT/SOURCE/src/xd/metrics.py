import numpy as np
import scipy
import skimage
import torch


def normalize(x):
    t = (x - np.min(x)) / (np.max(x) - np.min(x))
    return t


def normalize_sum_to_one(x):
    norm = normalize(x)
    return norm / norm.sum()


def feature_agreement(x, y, k=1638):
    _, i1 = torch.topk(torch.Tensor(x).flatten().abs(), k)
    _, i2 = torch.topk(torch.Tensor(y).flatten().abs(), k)
    i1 = set(torch.unique(i1).tolist())
    i2 = set(torch.unique(i2).tolist())
    intersect = len(i1.intersection(i2))
    return 1.0 * intersect / k


def sign_agreement(x, y, k=1638):
    x_flatten = x.flatten()
    y_flatten = y.flatten()
    x_topk = np.argpartition(np.absolute(x_flatten), -k)[-k:]
    y_topk = np.argpartition(np.absolute(y_flatten), -k)[-k:]

    intersect = np.intersect1d(x_topk, y_topk)
    same_sign = np.asarray(list(map(lambda ind: np.sign(
        x_flatten[ind]) == np.sign(y_flatten[ind]), intersect)))
    return same_sign.sum() * 1.0 / k


def rank_correlation(x, y):
    return scipy.stats.spearmanr(x.flatten().abs(), y.flatten().abs()).correlation


def ssim(x, y):
    return skimage.metrics.structural_similarity(normalize(np.abs(x)), normalize(np.abs(y)), data_range=1.0)
