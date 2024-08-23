from src.caketool.metric import gini


def test_gini_1():
    assert gini([1,1,1,0,0,0], [0.8, 0.8, 0.8, 0.3, 0.3, 0.3]) == 1

def test_gini_2():
    assert gini([1,1,1,0,0,0], [0.8, 0.8, 0.1, 0.3, 0.3, 0.3]) == 0.33333333333333326