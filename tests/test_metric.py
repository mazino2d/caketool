from caketool.metric import gini


def test_gini_1():
    assert gini([1, 1, 1, 0, 0, 0], [0.8, 0.8, 0.8, 0.3, 0.3, 0.3]) == 1


def test_gini_2():
    assert gini([1, 1, 1, 0, 0, 0], [0.8, 0.8, 0.1, 0.3, 0.3, 0.3]) == 0.33333333333333326


def test_gini_can_be_negative():
    # Inverted scores rank worse than random
    assert gini([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == -1.0
