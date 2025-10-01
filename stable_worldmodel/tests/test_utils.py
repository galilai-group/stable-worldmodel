import pytest

from stable_worldmodel.utils import get_in, flatten_dict


########################
## flatten_dict tests ##
########################


def test_flatten_dict_empty_dict():
    flatten_dict({}) == {}


def test_flatten_dict_single_level():
    assert flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_flatten_dict_nested_dict():
    assert flatten_dict({"a": {"b": 2}}) == {"a.b": 2}


def test_flatten_dict_information_loss():
    assert flatten_dict({"a": {"b": 2}, "a.b": 3}) == {"a.b": 3}


def test_flatten_dict_multiple_nested_levels():
    assert flatten_dict({"a": {"b": {"c": 3}}}) == {"a.b.c": 3}


def test_flatten_dict_other_separators():
    assert flatten_dict({"a": {"b": 2}}, sep="_") == {"a_b": 2}


def test_flatten_dict_parent_key():
    assert flatten_dict({"a": {"b": 2}}, parent_key="root") == {"root.a.b": 2}


def test_flatten_dict_mixed_types():
    assert flatten_dict({"a": {1: "string", (4, "5"): 2}}) == {
        "a.1": "string",
        "a.(4, '5')": 2,
    }


def test_flatten_dict_same_flatten():
    assert flatten_dict({"a": {"b": {"c": 3}}, "d": 4}) == flatten_dict(
        {"a": {"b.c": 3}, "d": 4}
    )


#################
## get_in test ##
#################


def test_get_in_existing_key_depth_one():
    assert get_in({"a": 2}, ["a"]) == 2


def test_get_in_empty_dict():
    with pytest.raises(KeyError):
        get_in({}, ["a"])


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
