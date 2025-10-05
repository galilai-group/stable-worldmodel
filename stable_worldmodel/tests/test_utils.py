import subprocess
from unittest.mock import MagicMock

import pytest

from stable_worldmodel.utils import flatten_dict, get_in, pretraining


#######################
## pretraining tests ##
#######################


def test_raises_when_script_not_exists(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: False)

    with pytest.raises(ValueError, match=r"does not exist"):
        pretraining("non_existent_script.py", "test_dataset", "test_model")


def test_pretraining_success(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    mock_run = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr("subprocess.run", mock_run)

    pretraining("fake_script.py", "test_dataset", "test_model", args="epochs=10")

    assert mock_run.called


def test_pretraining_exits_on_failure(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    mock_run = MagicMock(side_effect=subprocess.CalledProcessError(1, "cmd"))
    monkeypatch.setattr("subprocess.run", mock_run)

    with pytest.raises(SystemExit, match="1"):
        pretraining("fake_script.py", "test_dataset", "test_model", args="epochs=10")


def test_pretraining_parses_single_arg(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    mock_run = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_run)

    pretraining("fake_script.py", "test_dataset", "test_model", args="batch-size=32")

    cmd = mock_run.call_args[0][0]
    assert "fake_script.py" in cmd
    assert "batch-size=32" in cmd
    assert "++dump_object=True" in cmd  # default value
    assert "dataset_name=test_dataset" in cmd
    assert "output_model_name=test_model" in cmd
    assert mock_run.call_args[1]["check"] is True


def test_pretraining_parses_multiple_args(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    mock_run = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_run)

    pretraining("fake_script.py", "test_dataset", "test_model", args="batch-size=32 ++learning_rate=0.001")

    cmd = mock_run.call_args[0][0]
    assert "fake_script.py" in cmd
    assert "batch-size=32" in cmd
    assert "++learning_rate=0.001" in cmd
    assert "++dump_object=True" in cmd  # default value
    assert "dataset_name=test_dataset" in cmd
    assert "output_model_name=test_model" in cmd


def test_pretraining_with_empty_args(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    mock_run = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_run)

    pretraining("fake_script.py", "test_dataset", "test_model")

    cmd = mock_run.call_args[0][0]
    assert "fake_script.py" in cmd
    assert "++dump_object=True" in cmd  # default value
    assert "dataset_name=test_dataset" in cmd
    assert "output_model_name=test_model" in cmd


def test_pretraining_with_dump_object_false(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    mock_run = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_run)

    pretraining("fake_script.py", "test_dataset", "test_model", dump_object=False)

    cmd = mock_run.call_args[0][0]
    assert "fake_script.py" in cmd
    assert "++dump_object=False" in cmd
    assert "dataset_name=test_dataset" in cmd
    assert "output_model_name=test_model" in cmd


def test_pretraining_with_all_parameters(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    mock_run = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_run)

    pretraining("fake_script.py", "my_dataset", "my_model", dump_object=False, args="epochs=100")

    cmd = mock_run.call_args[0][0]
    assert "fake_script.py" in cmd
    assert "epochs=100" in cmd
    assert "++dump_object=False" in cmd
    assert "dataset_name=my_dataset" in cmd
    assert "output_model_name=my_model" in cmd


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
    assert flatten_dict({"a": {"b": {"c": 3}}, "d": 4}) == flatten_dict({"a": {"b.c": 3}, "d": 4})


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
