from src.caketool.utils.lib_utils import get_class
from os import PathLike

def test_get_PathLike_class():
    assert get_class("os.PathLike") == PathLike