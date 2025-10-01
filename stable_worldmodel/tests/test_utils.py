import pytest

from stable_worldmodel.utils import get_in


##

## get_in tests ##


def test_get_in_existing_key_depth_one():
    assert get_in({"a": 2}, ["a"]) == 2


def test_get_in_missing_key_depth_one():
    with pytest.raises(KeyError):
        get_in({"a": 1}, ["b"])


def test_get_in_empty_path():
    assert get_in({"a": 1}, []) == {"a": 1}


def test_get_in_existing_key_depth_two():
    assert get_in({"a": {"b": 3}}, ["a", "b"]) == 3


def test_get_in_missing_key_depth_two():
    with pytest.raises(KeyError):
        get_in({"a": {"b": 3}}, ["a", "c"])


def test_get_in_missing_intermediate_key_depth_two():
    with pytest.raises(KeyError):
        get_in({"a": {"b": 3}}, ["x", "b"])


def test_get_in_empty_key_depth_two():
    assert get_in({"a": {"b": 3}}, ["a"]) == {"b": 3}
