import numpy as np
from numba import jit
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re


@jit(nopython=True, parallel=True)
def numba_cluster_feat_mccs(cluster_id, feature_vecs, cluster_vec):
    """numba-accelerated matthews correlation coefficient calculation
    for finding base features most heavily correlated with cluster ids"""
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


def one_hot_encode(X):
    """one-hot encode a list of lists of data"""
    classes = sorted(list({item for sublist in X for item in sublist}))
    encoded = np.zeros((len(X), len(classes)), dtype=int)
    indices = [(i, classes.index(item)) for i, row in enumerate(X) for item in row]
    rows, cols = zip(*indices)
    encoded[rows, cols] = 1
    return encoded, classes


def cfeatures(group_ids, data, top_k=5, show_values=False):
    """Find the top k features most heavily correlated with each group id"""
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
        # top_vars = [reversed_vocab[i] for i in top_idxs]
        top_cluster_vars[cl] = top_vars
    return top_cluster_vars


def find_mwe(texts, n=10, min_df=5, max_df=0.9):
    """find the meaningful multi-word expressions in a list of texts

    Calculates bigram PMI scores for a list of texts and returns the top n MWEs

    Args:
        texts (list): a list of strings
        n (int, optional): the number of top MWEs to return. Defaults to 10.
        min_df (int, optional): the minimum document frequency for a term to be included. Defaults to 5.
        max_df (float, optional): the maximum document frequency for a term to be included. Defaults to 0.9.

    Returns:
        list: a list of the top n MWEs


    """
    # Tokenize to find unigrams and bigrams
    vectorizer = CountVectorizer(
        ngram_range=(1, 2), min_df=min_df, max_df=max_df, stop_words="english"
    )
    X = np.asarray(vectorizer.fit_transform(texts).sum(axis=0)).squeeze()

    def get_gram_indices(feature_names):
        pattern = re.compile("(?u)\\b\\w\\w+\\b")
        # pattern = re.compile(r'[a-zA-Z]{2,}')

        # splits = [pattern.findall(f) for f in feature_names]
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
    bi_feats = feats[bis]
    uni_feats = feats[unis]
    X_coc = X[bis]
    X_uni = X[unis]

    def get_bigram_split(bi_feats, uni_feats):
        if not isinstance(uni_feats, list):
            uni_feats = uni_feats.tolist()
        pattern = re.compile("(?u)\\b\\w\\w+\\b")
        # pattern = re.compile(r'[a-zA-Z]{2,}')

        uni_dict = {word: i for i, word in enumerate(uni_feats)}
        # splits = [pattern.findall(f) for f in bi_feats]
        splits = [[match.group() for match in pattern.finditer(f)] for f in bi_feats]

        # pairs = np.array([[uni_feats.index(s[0]),uni_feats.index(s[1])] for s in splits])
        pairs = np.array(
            [[uni_dict.get(s[0], -1), uni_dict.get(s[1], -1)] for s in splits]
        )
        return pairs

    bisplits = get_bigram_split(bi_feats, uni_feats)

    def pmi(X_coc, X_uni, bisplits):
        N = X_uni.sum()
        p, pxy = X_uni / N, X_coc / N
        ind_x = bisplits[:, 0]
        ind_y = bisplits[:, 1]
        px, py = p[ind_x], p[ind_y]
        pmi_ = np.log2(pxy / (px * py))
        return np.array(pmi_)

    PMI = pmi(X_coc, X_uni, bisplits)
    top_n = np.argsort(PMI)[-n:]

    top_n_bigrams = bi_feats[top_n]
    return top_n_bigrams
