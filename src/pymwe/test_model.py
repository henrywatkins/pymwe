from pymwe.model import *


def test_one_hot_encode():
    data = [["apple", "banana"], ["banana", "cherry"], ["apple", "cherry"]]
    encoded, classes = one_hot_encode(data)
    expected_encoded = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    expected_classes = ["apple", "banana", "cherry"]
    assert np.array_equal(encoded, expected_encoded)
    assert classes == expected_classes


def test_cfeatures():
    group_ids = [0, 1, 0, 1]
    data = [["apple", "banana"], ["banana", "cherry"], ["apple", "cherry"], ["banana"]]
    top_k = 2
    result = cfeatures(group_ids, data, top_k)
    assert 0 in result
    assert 1 in result
    assert len(result[0]) == top_k
    assert len(result[1]) == top_k


def test_find_mwe():
    texts = [
        "apple banana apple",
        "banana cherry banana",
        "apple cherry apple",
        "banana apple cherry",
    ]
    result = find_mwe(texts, n=2, min_df=1, max_df=0.9)
    assert len(result) == 2
    assert "apple banana" in result or "banana apple" in result
