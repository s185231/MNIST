import pytest
import warnings

def func1(x):
    assert x==3

def test_error_on_wrong_shape():
    warnings.warn(UserWarning('Test2'))
    return func1(3)

def func(x):
    if x == 3:
        raise ValueError("wrong number") 


def test_error_on_wrong_shape():
   with pytest.raises(ValueError, match='Test1 '):
        func(3)

