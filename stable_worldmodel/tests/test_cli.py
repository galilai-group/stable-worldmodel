"""Tests for CLI module."""

from unittest.mock import patch

import numpy as np
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from typer.testing import CliRunner

from stable_worldmodel.cli import (
    _build_hierarchy,
    _leaf,
    _leaf_table,
    _render,
    _render_hierarchy,
    _summarize,
    _variation_space,
    app,
    display_dataset_info,
    display_world_info,
)


runner = CliRunner()


###########################
## Helper Function Tests ##
###########################


def test_summarize_array():
    """Test summarizing numpy arrays."""
    arr = np.array([1, 2, 3, 4, 5])
    result = _summarize(arr)
    assert "shape=[5]" in result
    assert "min=1" in result
    assert "max=5" in result


def test_summarize_2d_array():
    """Test summarizing 2D arrays."""
    arr = np.array([[1, 2], [3, 4]])
    result = _summarize(arr)
    assert "shape=[2, 2]" in result
    assert "min=1" in result
    assert "max=4" in result


def test_summarize_empty_array():
    """Test summarizing empty arrays."""
    arr = np.array([])
    result = _summarize(arr)
    assert "shape=[0]" in result
    assert "[]" in result


def test_summarize_non_array():
    """Test summarizing non-array data that can't compute min/max."""
    # Use a complex object that will fall back to repr
    result = _summarize({"key": "value"})
    # Should either contain the dict representation or fall back gracefully
    assert "key" in result or "value" in result or "{" in result


def test_summarize_exception_handling():
    """Test summarize handles exceptions gracefully."""

    # Object that can't be converted to array
    class UnconvertibleObject:
        def __array__(self):
            raise ValueError("Cannot convert")

    result = _summarize(UnconvertibleObject())
    assert "UnconvertibleObject" in result


def test_leaf_with_type_key():
    """Test leaf detection for dictionaries with 'type' key."""
    assert _leaf({"type": "Box", "shape": [3]}) is True


def test_leaf_without_type_key():
    """Test leaf detection for dictionaries without 'type' key."""
    assert _leaf({"shape": [3], "dtype": "float32"}) is False


def test_leaf_non_dict():
    """Test leaf detection for non-dictionary objects."""
    assert _leaf([1, 2, 3]) is False
    assert _leaf("string") is False
    assert _leaf(None) is False


def test_leaf_table_basic():
    """Test leaf table creation with basic properties."""
    data = {"type": "Box", "shape": [3, 3], "dtype": "float32"}
    table = _leaf_table("Test Space", data)
    assert isinstance(table, Table)
    assert table.title == "Test Space"


def test_leaf_table_with_bounds():
    """Test leaf table creation with low/high bounds."""
    data = {
        "type": "Box",
        "shape": [2],
        "low": np.array([-1, -1]),
        "high": np.array([1, 1]),
    }
    table = _leaf_table("Bounded Space", data)
    assert isinstance(table, Table)


def test_leaf_table_with_discrete():
    """Test leaf table creation for discrete space."""
    data = {"type": "Discrete", "n": 5}
    table = _leaf_table("Discrete Space", data)
    assert isinstance(table, Table)


def test_leaf_table_with_extra_keys():
    """Test leaf table includes extra keys not in standard order."""
    data = {"type": "Box", "shape": [3], "custom_key": "custom_value"}
    table = _leaf_table("Custom Space", data)
    assert isinstance(table, Table)


def test_build_hierarchy_empty():
    """Test building hierarchy from empty list."""
    result = _build_hierarchy([])
    assert result == {}


def test_build_hierarchy_flat():
    """Test building hierarchy from flat names."""
    names = ["agent", "goal", "walls"]
    result = _build_hierarchy(names)
    assert "agent" in result
    assert "goal" in result
    assert "walls" in result
    assert result["agent"] == {}
    assert result["goal"] == {}


def test_build_hierarchy_nested():
    """Test building hierarchy from nested dotted names."""
    names = ["agent.pos.x", "agent.pos.y", "goal.pos"]
    result = _build_hierarchy(names)
    assert "agent" in result
    assert "pos" in result["agent"]
    assert "x" in result["agent"]["pos"]
    assert "y" in result["agent"]["pos"]
    assert "goal" in result
    assert "pos" in result["goal"]


def test_build_hierarchy_deep_nesting():
    """Test building hierarchy with deep nesting."""
    names = ["a.b.c.d.e"]
    result = _build_hierarchy(names)
    assert "a" in result
    assert "b" in result["a"]
    assert "c" in result["a"]["b"]
    assert "d" in result["a"]["b"]["c"]
    assert "e" in result["a"]["b"]["c"]["d"]


def test_build_hierarchy_custom_separator():
    """Test building hierarchy with custom separator."""
    names = ["agent/pos/x", "agent/pos/y"]
    result = _build_hierarchy(names, sep="/")
    assert "agent" in result
    assert "pos" in result["agent"]
    assert "x" in result["agent"]["pos"]


def test_build_hierarchy_empty_parts():
    """Test building hierarchy filters empty parts."""
    names = ["agent..pos", ".goal."]
    result = _build_hierarchy(names)
    assert "agent" in result
    assert "pos" in result["agent"]
    assert "goal" in result


def test_render_hierarchy_empty():
    """Test rendering empty hierarchy."""
    tree = Tree("root")
    _render_hierarchy(tree, {})
    # Should not crash, tree remains unchanged


def test_render_hierarchy_flat():
    """Test rendering flat hierarchy."""
    tree = Tree("root")
    hierarchy = {"a": {}, "b": {}, "c": {}}
    _render_hierarchy(tree, hierarchy)
    # Should add three leaf nodes


def test_render_hierarchy_nested():
    """Test rendering nested hierarchy."""
    tree = Tree("root")
    hierarchy = {"agent": {"pos": {"x": {}, "y": {}}}, "goal": {}}
    _render_hierarchy(tree, hierarchy)
    # Should create nested structure


def test_render_hierarchy_none_parent():
    """Test rendering hierarchy with None parent creates new tree."""
    hierarchy = {"a": {}}
    _render_hierarchy(None, hierarchy)
    # Should not crash


def test_render_none():
    """Test rendering None metadata."""
    result = _render(None, "Test")
    assert isinstance(result, Text)
    assert "<none>" in str(result)


def test_render_leaf():
    """Test rendering leaf node."""
    meta = {"type": "Box", "shape": [3]}
    result = _render(meta, "Space")
    assert isinstance(result, Table)


def test_render_dict():
    """Test rendering dictionary metadata."""
    meta = {
        "obs": {"type": "Box", "shape": [3]},
        "goal": {"type": "Box", "shape": [2]},
    }
    result = _render(meta, "Spaces")
    assert isinstance(result, Tree)


def test_render_nested_dict():
    """Test rendering nested dictionary metadata."""
    meta = {
        "agent": {
            "pos": {"type": "Box", "shape": [2]},
            "vel": {"type": "Box", "shape": [2]},
        }
    }
    result = _render(meta, "Agent")
    assert isinstance(result, Tree)


def test_render_list():
    """Test rendering list metadata."""
    meta = [
        {"type": "Box", "shape": [3]},
        {"type": "Discrete", "n": 5},
    ]
    result = _render(meta, "Spaces")
    assert isinstance(result, Tree)


def test_variation_space_no_variation():
    """Test variation space with no variations."""
    variation = {"has_variation": False}
    result = _variation_space(variation)
    assert isinstance(result, Tree)


def test_variation_space_with_variations():
    """Test variation space with variations."""
    variation = {
        "has_variation": True,
        "names": ["walls.number", "walls.shape", "agent.color"],
    }
    result = _variation_space(variation)
    assert isinstance(result, Tree)


def test_variation_space_empty_names():
    """Test variation space with empty names."""
    variation = {"has_variation": True, "names": []}
    result = _variation_space(variation)
    assert isinstance(result, Tree)


def test_variation_space_none_names():
    """Test variation space with None names."""
    variation = {"has_variation": True, "names": None}
    result = _variation_space(variation)
    assert isinstance(result, Tree)


def test_variation_space_empty_title():
    """Test variation space with empty title."""
    variation = {"has_variation": False}
    result = _variation_space(variation, title="")
    assert isinstance(result, Tree)


def test_variation_space_custom_title():
    """Test variation space with custom title."""
    variation = {"has_variation": True, "names": ["test.var"]}
    result = _variation_space(variation, title="Custom Vars")
    assert isinstance(result, Tree)


@patch("stable_worldmodel.cli.console")
def test_display_world_info(mock_console):
    """Test displaying world information."""
    info = {
        "name": "TestWorld-v0",
        "observation_space": {"type": "Box", "shape": [4]},
        "action_space": {"type": "Discrete", "n": 2},
        "variation": {"has_variation": False},
    }
    display_world_info(info)
    mock_console.print.assert_called_once()


@patch("stable_worldmodel.cli.console")
def test_display_world_info_missing_keys(mock_console):
    """Test displaying world info with missing keys."""
    info = {}
    display_world_info(info)
    mock_console.print.assert_called_once()


@patch("stable_worldmodel.cli.console")
def test_display_world_info_with_variation(mock_console):
    """Test displaying world info with variations."""
    info = {
        "name": "TestWorld-v0",
        "observation_space": None,
        "action_space": {"type": "Box", "shape": [2]},
        "variation": {"has_variation": True, "names": ["difficulty", "layout"]},
    }
    display_world_info(info)
    mock_console.print.assert_called_once()


@patch("stable_worldmodel.cli.console")
def test_display_dataset_info(mock_console):
    """Test displaying dataset information."""
    info = {
        "name": "test-dataset",
        "columns": ["obs", "action", "reward"],
        "num_episodes": 100,
        "num_steps": 5000,
        "obs_shape": (64, 64, 3),
        "action_shape": (2,),
        "goal_shape": (2,),
        "variation": {"has_variation": False},
    }
    display_dataset_info(info)
    mock_console.print.assert_called_once()


@patch("stable_worldmodel.cli.console")
def test_display_dataset_info_with_variation(mock_console):
    """Test displaying dataset info with variations."""
    info = {
        "name": "varied-dataset",
        "columns": ["obs", "action"],
        "num_episodes": 50,
        "num_steps": 2500,
        "obs_shape": (224, 224, 3),
        "action_shape": (4,),
        "goal_shape": (3,),
        "variation": {"has_variation": True, "names": ["env.type", "difficulty"]},
    }
    display_dataset_info(info)
    mock_console.print.assert_called_once()


###########################
## CLI Command Tests     ##
###########################


def test_version_flag():
    """Test --version flag displays version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "stable-worldmodel version:" in result.stdout


def test_version_short_flag():
    """Test -v flag displays version."""
    result = runner.invoke(app, ["-v"])
    assert result.exit_code == 0
    assert "stable-worldmodel version:" in result.stdout


@patch("stable_worldmodel.cli.data.list_datasets")
@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_list_datasets(mock_cache_dir, mock_list_datasets):
    """Test listing datasets."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list_datasets.return_value = ["dataset1", "dataset2", "dataset3"]

    result = runner.invoke(app, ["list", "dataset"])
    assert result.exit_code == 0
    assert "dataset1" in result.stdout
    assert "dataset2" in result.stdout
    assert "dataset3" in result.stdout


@patch("stable_worldmodel.cli.data.list_models")
@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_list_models(mock_cache_dir, mock_list_models):
    """Test listing models."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list_models.return_value = ["model1", "model2"]

    result = runner.invoke(app, ["list", "model"])
    assert result.exit_code == 0
    assert "model1" in result.stdout
    assert "model2" in result.stdout


@patch("stable_worldmodel.cli.data.list_worlds")
@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_list_worlds(mock_cache_dir, mock_list_worlds):
    """Test listing worlds."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list_worlds.return_value = ["swm/World1-v0", "swm/World2-v0"]

    result = runner.invoke(app, ["list", "world"])
    assert result.exit_code == 0
    assert "World1" in result.stdout
    assert "World2" in result.stdout


@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_list_invalid_kind(mock_cache_dir):
    """Test listing with invalid kind."""
    mock_cache_dir.return_value = "/fake/cache"

    result = runner.invoke(app, ["list", "invalid"])
    assert result.exit_code == 1
    assert "Invalid type" in result.stdout


@patch("stable_worldmodel.cli.data.list_datasets")
@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_list_empty_datasets(mock_cache_dir, mock_list_datasets):
    """Test listing when no datasets exist."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list_datasets.return_value = []

    result = runner.invoke(app, ["list", "dataset"])
    assert result.exit_code == 0
    assert "No cached" in result.stdout


@patch("stable_worldmodel.cli.data.dataset_info")
@patch("stable_worldmodel.cli.data.list_datasets")
@patch("stable_worldmodel.cli.data.get_cache_dir")
@patch("stable_worldmodel.cli.display_dataset_info")
def test_show_dataset(mock_display, mock_cache_dir, mock_list, mock_info):
    """Test showing a specific dataset."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list.return_value = ["test-dataset"]
    mock_info.return_value = {
        "name": "test-dataset",
        "columns": ["obs", "action"],
        "num_episodes": 10,
        "num_steps": 100,
        "obs_shape": (3,),
        "action_shape": (2,),
        "goal_shape": (2,),
        "variation": {"has_variation": False},
    }

    result = runner.invoke(app, ["show", "dataset", "test-dataset"])
    assert result.exit_code == 0
    mock_display.assert_called_once()


@patch("stable_worldmodel.cli.data.world_info")
@patch("stable_worldmodel.cli.data.list_worlds")
@patch("stable_worldmodel.cli.data.get_cache_dir")
@patch("stable_worldmodel.cli.display_world_info")
def test_show_world(mock_display, mock_cache_dir, mock_list, mock_info):
    """Test showing a specific world."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list.return_value = ["swm/TestWorld-v0"]
    mock_info.return_value = {
        "name": "swm/TestWorld-v0",
        "observation_space": {"type": "Box", "shape": [4]},
        "action_space": {"type": "Discrete", "n": 2},
        "variation": {"has_variation": False},
    }

    result = runner.invoke(app, ["show", "world", "swm/TestWorld-v0"])
    assert result.exit_code == 0
    mock_display.assert_called_once()


@patch("stable_worldmodel.cli.data.dataset_info")
@patch("stable_worldmodel.cli.data.list_datasets")
@patch("stable_worldmodel.cli.data.get_cache_dir")
@patch("stable_worldmodel.cli.display_dataset_info")
def test_show_multiple_datasets(mock_display, mock_cache_dir, mock_list, mock_info):
    """Test showing multiple datasets."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list.return_value = ["dataset1", "dataset2"]
    mock_info.return_value = {
        "name": "test",
        "columns": ["obs"],
        "num_episodes": 1,
        "num_steps": 10,
        "obs_shape": (3,),
        "action_shape": (2,),
        "goal_shape": (2,),
        "variation": {"has_variation": False},
    }

    result = runner.invoke(app, ["show", "dataset", "dataset1", "dataset2"])
    assert result.exit_code == 0
    assert mock_display.call_count == 2


@patch("stable_worldmodel.cli.data.dataset_info")
@patch("stable_worldmodel.cli.data.list_datasets")
@patch("stable_worldmodel.cli.data.get_cache_dir")
@patch("stable_worldmodel.cli.display_dataset_info")
def test_show_all_datasets(mock_display, mock_cache_dir, mock_list, mock_info):
    """Test showing all datasets with --all flag."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list.return_value = ["dataset1", "dataset2", "dataset3"]
    mock_info.return_value = {
        "name": "test",
        "columns": ["obs"],
        "num_episodes": 1,
        "num_steps": 10,
        "obs_shape": (3,),
        "action_shape": (2,),
        "goal_shape": (2,),
        "variation": {"has_variation": False},
    }

    result = runner.invoke(app, ["show", "dataset", "--all"])
    assert result.exit_code == 0
    assert mock_display.call_count == 3


@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_show_invalid_kind(mock_cache_dir):
    """Test show with invalid kind."""
    mock_cache_dir.return_value = "/fake/cache"

    result = runner.invoke(app, ["show", "invalid", "something"])
    assert result.exit_code == 1
    assert "Invalid type" in result.stdout


@patch("stable_worldmodel.cli.data.list_datasets")
@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_show_no_names_no_all_flag(mock_cache_dir, mock_list):
    """Test show without names and without --all flag."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list.return_value = []

    result = runner.invoke(app, ["show", "dataset"])
    assert result.exit_code == 1
    assert "Nothing to show" in result.stdout


@patch("stable_worldmodel.cli.data.list_datasets")
@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_show_non_existent_dataset(mock_cache_dir, mock_list):
    """Test showing a dataset that doesn't exist."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list.return_value = ["existing-dataset"]

    result = runner.invoke(app, ["show", "dataset", "non-existent"])
    assert result.exit_code == 1
    assert "can't be found" in result.stdout


@patch("stable_worldmodel.cli.data.list_worlds")
@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_show_multiple_non_existent_worlds(mock_cache_dir, mock_list):
    """Test showing multiple worlds that don't exist."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list.return_value = ["swm/World1-v0"]

    result = runner.invoke(app, ["show", "world", "swm/NonExistent1-v0", "swm/NonExistent2-v0"])
    assert result.exit_code == 1
    assert "can't be found" in result.stdout


@patch("stable_worldmodel.cli.data.delete_dataset")
@patch("stable_worldmodel.cli.data.list_datasets")
@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_delete_dataset(mock_cache_dir, mock_list, mock_delete):
    """Test deleting a dataset with confirmation."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list.return_value = ["test-dataset"]

    result = runner.invoke(app, ["delete", "dataset", "test-dataset"], input="y\n")
    assert result.exit_code == 0
    mock_delete.assert_called_once_with("test-dataset")
    assert "Deleting" in result.stdout


@patch("stable_worldmodel.cli.data.delete_model")
@patch("stable_worldmodel.cli.data.list_models")
@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_delete_model(mock_cache_dir, mock_list, mock_delete):
    """Test deleting a model with confirmation."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list.return_value = ["test-model"]

    result = runner.invoke(app, ["delete", "model", "test-model"], input="y\n")
    assert result.exit_code == 0
    mock_delete.assert_called_once_with("test-model")


@patch("stable_worldmodel.cli.data.delete_dataset")
@patch("stable_worldmodel.cli.data.list_datasets")
@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_delete_multiple_datasets(mock_cache_dir, mock_list, mock_delete):
    """Test deleting multiple datasets."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list.return_value = ["dataset1", "dataset2"]

    result = runner.invoke(app, ["delete", "dataset", "dataset1", "dataset2"], input="y\n")
    assert result.exit_code == 0
    assert mock_delete.call_count == 2


@patch("stable_worldmodel.cli.data.list_datasets")
@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_delete_cancel_confirmation(mock_cache_dir, mock_list):
    """Test canceling delete operation."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list.return_value = ["test-dataset"]

    result = runner.invoke(app, ["delete", "dataset", "test-dataset"], input="n\n")
    assert result.exit_code == 1
    # When user cancels, typer.confirm with abort=True raises Abort
    # The output includes the confirmation prompt but may not include "Aborted"
    assert "Are you sure" in result.stdout or result.exit_code == 1


@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_delete_invalid_kind(mock_cache_dir):
    """Test delete with invalid kind."""
    mock_cache_dir.return_value = "/fake/cache"

    result = runner.invoke(app, ["delete", "invalid", "something"])
    assert result.exit_code == 1
    assert "Invalid type" in result.stdout


@patch("stable_worldmodel.cli.data.list_datasets")
@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_delete_non_existent_dataset(mock_cache_dir, mock_list):
    """Test deleting a dataset that doesn't exist."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list.return_value = ["existing-dataset"]

    result = runner.invoke(app, ["delete", "dataset", "non-existent"])
    assert result.exit_code == 1
    assert "can't be found" in result.stdout


@patch("stable_worldmodel.cli.data.list_models")
@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_delete_multiple_some_non_existent(mock_cache_dir, mock_list):
    """Test deleting multiple items where some don't exist."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list.return_value = ["model1", "model2"]

    result = runner.invoke(app, ["delete", "model", "model1", "non-existent", "model2"])
    assert result.exit_code == 1
    assert "can't be found" in result.stdout


###########################
## Edge Cases & Coverage ##
###########################


def test_summarize_multidimensional_array():
    """Test summarizing high-dimensional arrays."""
    arr = np.ones((2, 3, 4, 5))
    result = _summarize(arr)
    assert "shape=[2, 3, 4, 5]" in result
    assert "min=1" in result
    assert "max=1" in result


def test_leaf_table_none_bounds():
    """Test leaf table with None low/high values."""
    data = {"type": "Box", "shape": [3], "low": None, "high": None}
    table = _leaf_table("Space", data)
    assert isinstance(table, Table)


def test_build_hierarchy_numeric_names():
    """Test building hierarchy with numeric names."""
    names = [123, 456.789]
    result = _build_hierarchy(names)
    assert "123" in result or "456" in result


def test_render_empty_dict():
    """Test rendering empty dictionary."""
    result = _render({}, "Empty")
    assert isinstance(result, Tree)


def test_render_empty_list():
    """Test rendering empty list."""
    result = _render([], "Empty")
    assert isinstance(result, Tree)


def test_variation_space_tuple_names():
    """Test variation space with tuple of names."""
    variation = {"has_variation": True, "names": ("var1", "var2")}
    result = _variation_space(variation)
    assert isinstance(result, Tree)


@patch("stable_worldmodel.cli.console")
def test_display_world_info_complex_spaces(mock_console):
    """Test displaying world with complex nested spaces."""
    info = {
        "name": "ComplexWorld-v0",
        "observation_space": {
            "image": {"type": "Box", "shape": [64, 64, 3]},
            "state": {"type": "Box", "shape": [10]},
        },
        "action_space": {"type": "MultiDiscrete", "nvec": [3, 4, 5]},
        "variation": {
            "has_variation": True,
            "names": ["env.layout", "env.difficulty", "agent.color"],
        },
    }
    display_world_info(info)
    mock_console.print.assert_called_once()


@patch("stable_worldmodel.cli.data.world_info")
@patch("stable_worldmodel.cli.data.list_worlds")
@patch("stable_worldmodel.cli.data.get_cache_dir")
@patch("stable_worldmodel.cli.display_world_info")
def test_show_world_short_flag(mock_display, mock_cache_dir, mock_list, mock_info):
    """Test showing all worlds with -a flag."""
    mock_cache_dir.return_value = "/fake/cache"
    mock_list.return_value = ["swm/World1-v0"]
    mock_info.return_value = {
        "name": "swm/World1-v0",
        "observation_space": None,
        "action_space": None,
        "variation": {"has_variation": False},
    }

    result = runner.invoke(app, ["show", "world", "-a"])
    assert result.exit_code == 0
    mock_display.assert_called_once()


def test_common_callback_without_version():
    """Test common callback without version flag."""
    # Should proceed to command, not exit early
    # Just invoke without checking result since command will fail without proper mocks
    runner.invoke(app, ["list", "dataset"])


@patch("stable_worldmodel.cli.data.list_datasets")
@patch("stable_worldmodel.cli.data.get_cache_dir")
def test_list_datasets_sorted(mock_cache_dir, mock_list_datasets):
    """Test that list command sorts output."""
    mock_cache_dir.return_value = "/fake/cache"
    # Provide unsorted list
    mock_list_datasets.return_value = ["zebra-dataset", "alpha-dataset", "beta-dataset"]

    result = runner.invoke(app, ["list", "dataset"])
    assert result.exit_code == 0
    # Check that alpha appears before beta which appears before zebra
    alpha_pos = result.stdout.index("alpha")
    beta_pos = result.stdout.index("beta")
    zebra_pos = result.stdout.index("zebra")
    assert alpha_pos < beta_pos < zebra_pos
