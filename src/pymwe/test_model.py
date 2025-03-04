import numpy as np
import pytest

from pymwe.model import (cfeatures, find_mwe, numba_cluster_feat_mccs,
                         one_hot_encode)


def test_one_hot_encode():
    """Test one-hot encoding functionality."""
    data = [["apple", "banana"], ["banana", "cherry"], ["apple", "cherry"]]
    encoded, classes = one_hot_encode(data)
    expected_encoded = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    expected_classes = ["apple", "banana", "cherry"]
    assert np.array_equal(encoded, expected_encoded)
    assert classes == expected_classes


def test_cfeatures():
    """Test correlation features calculation."""
    group_ids = [0, 1, 0, 1]
    data = [["apple", "banana"], ["banana", "cherry"], ["apple", "cherry"], ["banana"]]
    top_k = 2
    result = cfeatures(group_ids, data, top_k)
    assert 0 in result
    assert 1 in result
    assert len(result[0]) == top_k
    assert len(result[1]) == top_k

    # Test with show_values=True
    result_with_values = cfeatures(group_ids, data, top_k, show_values=True)
    for cluster in result_with_values:
        for feature in result_with_values[cluster]:
            assert isinstance(feature, tuple)
            assert len(feature) == 2
            assert isinstance(feature[1], float)


def test_numba_cluster_feat_mccs():
    """Test the Matthews correlation coefficient calculation."""
    # Simple test case
    feature_vecs = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float64)
    cluster_vec = np.array([0, 1, 0, 1])

    # Test for cluster 0
    mccs0 = numba_cluster_feat_mccs(0, feature_vecs, cluster_vec)
    assert mccs0.shape == (2,)
    assert mccs0[0] > 0  # Positive correlation with first feature
    assert mccs0[1] < 0  # Negative correlation with second feature

    # Test for cluster 1
    mccs1 = numba_cluster_feat_mccs(1, feature_vecs, cluster_vec)
    assert mccs1.shape == (2,)
    assert mccs1[0] < 0  # Negative correlation with first feature
    assert mccs1[1] > 0  # Positive correlation with second feature


def test_find_mwe():
    """Test finding multi-word expressions."""
    texts = [
        "apple banana apple",
        "banana cherry banana",
        "apple cherry apple",
        "banana apple cherry",
    ]
    result = find_mwe(texts, n=2, min_df=1, max_df=0.9)
    assert len(result) == 2
    assert "apple banana" in result or "banana apple" in result

    # Test with different parameters
    result_small = find_mwe(texts, n=1, min_df=1, max_df=0.9)
    assert len(result_small) == 1

    # Test empty input
    empty_result = find_mwe([], n=2, min_df=1, max_df=0.9)
    assert len(empty_result) == 0

    # Test with single word texts
    single_words = ["apple", "banana", "cherry", "date"]
    single_result = find_mwe(single_words, n=2, min_df=1, max_df=0.9)
    assert len(single_result) <= 2  # May find fewer MWEs than requested


def test_find_mwe_edge_cases():
    """Test edge cases for find_mwe function."""
    # Test with repeated words
    texts = ["apple apple apple apple", "banana banana banana"]
    result = find_mwe(texts, n=1, min_df=1, max_df=1.0)
    assert len(result) <= 1  # Might find no meaningful bigrams

    # Test with uncommon words
    rare_texts = [
        "uncommon phrase here",
        "another uncommon phrase",
        "third uncommon example",
    ]
    # Should still work with low min_df
    result = find_mwe(rare_texts, n=2, min_df=1, max_df=1.0)
    assert len(result) <= 2
