"""Module for multi-word expression extraction and feature correlation analysis."""

import re
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
from numba import jit
from sklearn.feature_extraction.text import CountVectorizer


@jit(nopython=True, parallel=True)
def numba_cluster_feat_mccs(
    cluster_id: int, feature_vecs: np.ndarray, cluster_vec: np.ndarray
) -> np.ndarray:
    """Calculate Matthews correlation coefficient for features with a cluster.

    A numba-accelerated implementation of Matthews correlation coefficient
    calculation for finding base features most heavily correlated with cluster IDs.

    Args:
        cluster_id: The ID of the cluster to calculate MCCs for
        feature_vecs: Matrix of feature vectors (samples × features)
        cluster_vec: Vector of cluster IDs for each sample

    Returns:
        Array of Matthews correlation coefficients for each feature
    """
    mccs = []
    for feature_idx in range(feature_vecs.shape[1]):
        in_cluster = cluster_vec == cluster_id
        has_feature = feature_vecs[:, feature_idx] > 0.0
        tp = np.logical_and(in_cluster, has_feature).sum()
        fp = np.logical_and(np.logical_not(in_cluster), has_feature).sum()
        tn = np.logical_and(
            np.logical_not(in_cluster), np.logical_not(has_feature)
        ).sum()
        fn = np.logical_and(in_cluster, np.logical_not(has_feature)).sum()
        num = tp * tn - fp * fn
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        MCC = num / denom
        mccs.append(MCC)
    mccs = np.array(mccs)
    return mccs


def one_hot_encode(X: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
    """One-hot encode a list of lists of data.

    Args:
        X: List of lists of strings to encode

    Returns:
        Tuple containing:
            - Encoded matrix as a numpy array (samples × features)
            - List of class labels (feature names)
    """
    classes = sorted(list({item for sublist in X for item in sublist}))
    encoded = np.zeros((len(X), len(classes)), dtype=int)
    indices = [(i, classes.index(item)) for i, row in enumerate(X) for item in row]
    if not indices:
        return encoded, classes
    rows, cols = zip(*indices)
    encoded[rows, cols] = 1
    return encoded, classes


def cfeatures(
    group_ids: List[int],
    data: List[List[str]],
    top_k: int = 5,
    show_values: bool = False,
) -> Dict[int, Union[List[str], List[Tuple[str, float]]]]:
    """Find the top k features most heavily correlated with each group id.

    Args:
        group_ids: List of group/cluster IDs for each data point
        data: List of lists of features for each data point
        top_k: Number of top correlated features to return
        show_values: If True, return tuples of (feature, correlation_value)

    Returns:
        Dictionary mapping group IDs to lists of their top k correlated features
    """
    vecs, vocab = one_hot_encode(data)
    cluster_labels = np.array(group_ids)
    cluster_mccs = {}
    for cl in np.unique(cluster_labels):
        mccs = numba_cluster_feat_mccs(cl, vecs, cluster_labels)
        cluster_mccs[cl] = mccs
    reversed_vocab = dict(enumerate(vocab))
    top_cluster_vars = {}
    for cl in cluster_mccs.keys():
        top_idxs = np.argsort(cluster_mccs[cl])[-top_k:]
        if show_values:
            top_vars = [(reversed_vocab[i], cluster_mccs[cl][i]) for i in top_idxs]
        else:
            top_vars = [reversed_vocab[i] for i in top_idxs]
        top_cluster_vars[cl] = top_vars
    return top_cluster_vars


def find_mwe(
    texts: List[str], n: int = 10, min_df: int = 5, max_df: float = 0.9
) -> np.ndarray:
    """Find the meaningful multi-word expressions in a list of texts.

    Calculates bigram PMI (Pointwise Mutual Information) scores for a list
    of texts and returns the top n MWEs.

    Args:
        texts: A list of strings, each representing a document
        n: The number of top MWEs to return
        min_df: The minimum document frequency for a term to be included
        max_df: The maximum document frequency for a term to be included

    Returns:
        A numpy array of strings containing the top n MWEs
    """
    if not texts:
        return np.array([])

    # Tokenize to find unigrams and bigrams
    try:
        vectorizer = CountVectorizer(
            ngram_range=(1, 2), min_df=min_df, max_df=max_df, stop_words="english"
        )
        X = np.asarray(vectorizer.fit_transform(texts).sum(axis=0)).squeeze()
    except ValueError:
        # Handle case where no features meet the criteria
        return np.array([])

    def get_gram_indices(feature_names: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split feature names into unigrams and bigrams based on word count."""
        pattern = re.compile(r"(?u)\b\w\w+\b")
        splits = [
            [match.group() for match in pattern.finditer(f)] for f in feature_names
        ]

        unigrams = []
        bigrams = []
        for i, s in enumerate(splits):
            if len(s) == 1:
                unigrams.append(i)
            else:
                bigrams.append(i)

        return np.array(unigrams), np.array(bigrams)

    feats = vectorizer.get_feature_names_out()
    unis, bis = get_gram_indices(feats)

    if len(bis) == 0:
        # No bigrams found
        return np.array([])

    bi_feats = feats[bis]
    uni_feats = feats[unis]
    X_coc = X[bis]
    X_uni = X[unis]

    def get_bigram_split(bi_feats: np.ndarray, uni_feats: np.ndarray) -> np.ndarray:
        """Split bigram features into pairs of unigram indices."""
        if not isinstance(uni_feats, list):
            uni_feats = uni_feats.tolist()
        pattern = re.compile(r"(?u)\b\w\w+\b")

        uni_dict = {word: i for i, word in enumerate(uni_feats)}
        splits = [[match.group() for match in pattern.finditer(f)] for f in bi_feats]

        pairs = np.array(
            [[uni_dict.get(s[0], -1), uni_dict.get(s[1], -1)] for s in splits]
        )
        return pairs

    bisplits = get_bigram_split(bi_feats, uni_feats)

    def pmi(X_coc: np.ndarray, X_uni: np.ndarray, bisplits: np.ndarray) -> np.ndarray:
        """Calculate Pointwise Mutual Information for bigrams."""
        N = X_uni.sum()
        if N == 0:
            return np.zeros(len(X_coc))

        p, pxy = X_uni / N, X_coc / N
        ind_x = bisplits[:, 0]
        ind_y = bisplits[:, 1]
        px, py = p[ind_x], p[ind_y]
        # Avoid division by zero
        valid_indices = (px * py) > 0
        pmi_ = np.zeros(len(pxy))
        pmi_[valid_indices] = np.log2(
            pxy[valid_indices] / (px[valid_indices] * py[valid_indices])
        )
        return np.array(pmi_)

    PMI = pmi(X_coc, X_uni, bisplits)

    # Take at most n bigrams (or fewer if not enough are available)
    n_to_take = min(n, len(PMI))
    if n_to_take == 0:
        return np.array([])

    top_n = np.argsort(PMI)[-n_to_take:]
    top_n_bigrams = bi_feats[top_n]
    return top_n_bigrams
