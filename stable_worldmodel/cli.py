"""Command-line interface for stable-worldmodel.

This module provides CLI commands for managing and inspecting stable-worldmodel
resources including datasets, models, and world environments. The CLI uses the
Typer framework with Rich formatting for an enhanced terminal experience.

Available commands:
    - list: List cached models, datasets, or worlds
    - show: Display detailed information about datasets or worlds
    - delete: Remove models or datasets from cache

The CLI can be invoked via:
    - `stable-worldmodel <command>`
    - `swm <command>`
    - `python -m stable_worldmodel.cli <command>`

Typical usage examples:

    List all cached datasets::

        $ swm list dataset

    Show information about a specific world::

        $ swm show world swm/SimplePointMaze-v0

    Delete a cached dataset::

        $ swm delete dataset my-dataset

    Show version::

        $ swm --version
"""

from typing import Annotated, Any

import numpy as np
import typer
from rich import box, print
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from stable_worldmodel import data

from . import __version__


console = Console()


def _summarize(x: Any) -> str:
    """Summarize array-like data with shape and value range.

    Attempts to convert input to a numpy array and returns a string describing
    its shape, minimum, and maximum values. Falls back to repr() for non-array data.

    Args:
        x (Any): Data to summarize. Can be any type, but works best with array-like data.

    Returns:
        str: Summary string with shape and range, or repr() if conversion fails.

    Example:
        >>> arr = np.array([1, 2, 3, 4, 5])
        >>> _summarize(arr)
        'shape=[5], min=1, max=5'
    """
    try:
        a = np.asarray(x)
    except Exception:
        return repr(x)
    return f"shape={list(a.shape)}, min={a.min()}, max={a.max()}" if a.size else f"[] shape={list(a.shape)}"


def _leaf(m: Any) -> bool:
    """Check if a dictionary represents a leaf node in the space hierarchy.

    A leaf node is identified as a dictionary containing a 'type' key, which
    indicates it's a terminal space definition rather than a nested structure.

    Args:
        m (Any): Object to check, typically a dictionary.

    Returns:
        bool: True if m is a dict with 'type' key, False otherwise.
    """
    return isinstance(m, dict) and "type" in m


def _leaf_table(title: str, m: dict[str, Any]) -> Table:
    """Create a Rich table displaying leaf node space properties.

    Generates a formatted table showing space attributes like type, shape, dtype,
    etc. Properties are displayed in a specific order with special formatting for
    certain keys.

    Args:
        title (str): Table title, typically the space name.
        m (Dict[str, Any]): Dictionary containing space properties.

    Returns:
        Table: A Rich Table object formatted for display.

    Note:
        Standard keys (type, shape, dtype, n, low, high) are displayed first
        in that order, followed by any additional keys alphabetically.
    """
    t = Table(
        title=title,
        title_style="bold yellow",
        title_justify="left",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_header=False,
        pad_edge=False,
    )
    t.add_column("k", style="bold cyan", no_wrap=True)
    t.add_column("v")
    order = ["type", "shape", "dtype", "n", "low", "high"]
    for k in order:
        if k in m:
            v = _summarize(m[k]) if k in ("low", "high") and m[k] is not None else m[k]
            t.add_row(k, str(v))
    for k, v in m.items():
        if k not in order:
            t.add_row(k, str(v))
    return t


def _build_hierarchy(flat_names, sep="."):
    """Build a nested dictionary hierarchy from flat dotted names.

    Converts a list of dot-separated names into a nested dictionary structure,
    useful for displaying hierarchical space names in tree format.

    Args:
        flat_names: Iterable of string names with separator-delimited hierarchy.
        sep (str, optional): Separator character for splitting names. Defaults to ".".

    Returns:
        Dict[str, Dict]: Nested dictionary representing the hierarchy.

    Example:
        >>> names = ["agent.pos.x", "agent.pos.y", "goal.pos"]
        >>> _build_hierarchy(names)
        {'agent': {'pos': {'x': {}, 'y': {}}}, 'goal': {'pos': {}}}
    """
    root: dict[str, dict] = {}
    if not flat_names:
        return root
    for raw in flat_names:
        parts = [p for p in str(raw).split(sep) if p]
        cur = root
        for p in parts:
            cur = cur.setdefault(p, {})
    return root


def _render_hierarchy(parent: Tree, d: dict[str, dict]):
    """Recursively render a hierarchical dictionary as a Rich Tree.

    Traverses the nested dictionary and adds nodes to the Rich Tree, with
    different styling for leaf vs. non-leaf nodes.

    Args:
        parent (Tree): Parent Rich Tree node to add children to. Can be None
            for root level.
        d (Dict[str, Dict]): Nested dictionary representing the hierarchy.

    Note:
        Modifies the parent Tree in-place by adding child nodes.
    """
    for k in sorted(d):
        is_non_leaf = bool(d[k])
        label = Text(k, style="bold cyan") if is_non_leaf else Text(k)
        child = parent.add(label) if parent else Tree(label)
        if is_non_leaf:
            _render_hierarchy(child, d[k])


def _render(meta: dict[str, Any] | list[Any] | None, label: str):
    """Render space metadata as a Rich Tree or Table.

    Creates an appropriate Rich display object (Tree or Table) for the given
    metadata structure. Handles nested dictionaries, lists, and leaf nodes.

    Args:
        meta (Union[Dict[str, Any], List[Any], None]): Space metadata to render.
            Can be a nested dictionary, list, or None.
        label (str): Label for the root of this metadata section.

    Returns:
        Union[Tree, Table, Text]: Rich display object representing the metadata.
            Returns Text for None, Table for leaf nodes, Tree for nested structures.

    Note:
        This function is called recursively to handle nested space definitions.
    """
    if meta is None:
        return Text(f"{label}: <none>", style="italic")
    if _leaf(meta):
        return _leaf_table(label, meta)
    tree = Tree(Text(label, style="bold yellow"))
    if isinstance(meta, dict):
        for k, v in meta.items():
            tree.add(_leaf_table(k, v) if _leaf(v) else _render(v, k))
    else:
        for i, v in enumerate(meta):
            title = f"#{i}"
            tree.add(_leaf_table(title, v) if _leaf(v) else _render(v, title))
    return tree


def _variation_space(variation: dict[str, Any], title: str = "Variation Space"):
    """Create a Rich Tree displaying variation space information.

    Renders the variation space structure showing available environment variations
    and their hierarchical organization.

    Args:
        variation (Dict[str, Any]): Dictionary containing variation space metadata.
            Expected keys include 'has_variation' and 'names'.
        title (str, optional): Title for the variation space tree. Defaults to
            "Variation Space". Empty string creates a minimal display.

    Returns:
        Tree: Rich Tree object displaying the variation space structure.

    Note:
        If the environment has no variations, displays a message instead of
        the hierarchical structure.
    """
    vroot = Tree(Text(title, style="bold yellow"))
    void_title = title == ""

    # small facts table (aligned titles)
    if not variation.get("has_variation"):
        text = "There are no variations 🙁"
        if void_title:
            vroot.label = text
        else:
            vroot.add(text)

    else:
        # hierarchical names
        names = variation.get("names") or []
        if isinstance(names, (list | tuple)) and names:
            tree_dict = _build_hierarchy(names, sep=".")
            _render_hierarchy(vroot, tree_dict)
        else:
            vroot.add(Text("names: —", style="dim"))

    return vroot


def display_world_info(info: dict[str, Any]) -> None:
    """Display world environment information in a formatted panel.

    Prints a Rich panel showing the world's observation space, action space,
    and variation space in a hierarchical, color-coded format.

    Args:
        info (Dict[str, Any]): World metadata dictionary containing keys:
            - 'name': World environment name
            - 'observation_space': Observation space structure
            - 'action_space': Action space structure
            - 'variation': Variation space structure

    Example:
        >>> from stable_worldmodel import data
        >>> info = data.world_info("swm/SimplePointMaze-v0")
        >>> display_world_info(info)
    """
    root = Tree(Text(f"World: {info.get('name', '<unknown>')}", style="bold green"))
    root.add(_render(info.get("observation_space"), "Extra Observation Space"))
    root.add(_render(info.get("action_space"), "Action Space"))
    root.add(_variation_space(info.get("variation", {})))

    console.print(
        Panel(
            root,
            title="[b]World Info[/b]",
            border_style="green",
            padding=(1, 2),
            title_align="center",
        )
    )


def display_dataset_info(info: dict[str, Any]) -> None:
    """Display dataset information in a formatted panel with tables.

    Prints a Rich panel showing dataset metadata including columns, episode count,
    shapes, and variation information in a two-table layout.

    Args:
        info (Dict[str, Any]): Dataset metadata dictionary containing keys:
            - 'name': Dataset name
            - 'columns': List of data columns
            - 'num_episodes': Total number of episodes
            - 'num_steps': Total number of steps
            - 'obs_shape': Observation tensor shape
            - 'action_shape': Action tensor shape
            - 'goal_shape': Goal tensor shape
            - 'variation': Variation space structure

    Example:
        >>> from stable_worldmodel import data
        >>> info = data.dataset_info("simple-pointmaze")
        >>> display_dataset_info(info)
    """
    top_table = Table(
        title=f"Dataset: [bold cyan]{info['name']}[/bold cyan]",
        box=box.SIMPLE_HEAVY,
        show_header=False,
        pad_edge=False,
    )

    top_table.add_column("Key", style="bold yellow", no_wrap=True)
    top_table.add_column("Value", style="white")
    top_table.add_row("Columns", ", ".join(info["columns"]))

    separator = Rule(characters="·", style="grey62")

    bottom_table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=False,
        pad_edge=False,
    )
    bottom_table.add_column("Key", style="bold yellow", no_wrap=True)
    bottom_table.add_column("Value", style="white")

    bottom_table.add_row("Episodes", str(info["num_episodes"]))
    bottom_table.add_row("Total Steps", str(info["num_steps"]))
    bottom_table.add_row("Obs Shape", str(info["obs_shape"]))
    bottom_table.add_row("Action Shape", str(info["action_shape"]))
    bottom_table.add_row("Goal Shape", str(info["goal_shape"]))
    bottom_table.add_row("Variation", _variation_space(info["variation"], title=""))

    group = Group(top_table, separator, bottom_table)

    console.print(
        Panel(
            group,
            border_style="cyan",
            padding=(1, 2),
        )
    )


##############
##   APP    ##
##############


app = typer.Typer()


def _version_callback(value: bool):
    """Display stable-worldmodel version and exit.

    Typer callback function that prints the installed version when the
    --version flag is used.

    Args:
        value (bool): True if --version flag was provided.

    Raises:
        typer.Exit: Always exits after displaying version.
    """
    if value:
        typer.echo(f"stable-worldmodel version: {__version__}")
        raise typer.Exit()


@app.callback()
def common(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            callback=_version_callback,
            help="Show installed stable-wordlnodel version.",
        ),
    ] = None,
):
    """Common options for all stable-worldmodel commands.

    This callback provides global options that apply to all CLI commands.
    Currently only supports the --version flag.

    Args:
        version (Optional[bool], optional): If True, display version and exit.
            Defaults to None.
    """
    pass


@app.command("list")
def list_cmd(
    kind: Annotated[str, typer.Argument(help="Type to list: 'model', 'dataset' or 'world'")],
):
    """List cached stable-worldmodel resources.

    Displays a table of all cached models, datasets, or worlds stored in the
    stable-worldmodel cache directory. Useful for seeing what resources are
    available locally.

    Args:
        kind (str): Type of resource to list. Must be one of:
            - 'model': List cached world models
            - 'dataset': List cached datasets
            - 'world': List registered world environments

    Raises:
        typer.Abort: If kind is not 'model', 'dataset', or 'world'.

    Example:
        List all cached datasets::

            $ swm list dataset

        List all available worlds::

            $ swm list world
    """
    cache_dir = data.get_cache_dir()

    if kind == "dataset":
        cached_items = data.list_datasets()
    elif kind == "model":
        cached_items = data.list_models()
    elif kind == "world":
        cached_items = data.list_worlds()
    else:
        print("[red]Invalid type: must be 'model', 'dataset' or 'world'[/red]")
        raise typer.Abort()

    if not cached_items:
        print(f"[yellow]No cached {kind}s found in {cache_dir}[/yellow]")
        return

    table = Table(
        title=f"Cached {kind}s in [dim]{cache_dir}[/dim]",
        header_style="bold cyan",
        box=box.SIMPLE,
    )
    table.add_column("Name", style="green", no_wrap=True)

    for item in sorted(cached_items):
        table.add_row(item)

    print(table)


@app.command()
def show(
    kind: Annotated[str, typer.Argument(help="Type to show: 'dataset' or 'world'")],
    names: Annotated[list[str] | None, typer.Argument(help="Names of worlds or datasets to show")] = None,
    all: Annotated[
        bool,
        typer.Option("--all", "-a", help="Show all cached datasets/worlds", is_flag=True),
    ] = False,
):
    """Show detailed information about datasets or worlds.

    Displays comprehensive information about specified datasets or world environments
    including their structure, spaces, and variations. Information is presented in
    formatted panels with hierarchical trees and tables.

    Args:
        kind (str): Type of resource to show. Must be either:
            - 'dataset': Show dataset information
            - 'world': Show world environment information
        names (Optional[List[str]], optional): Specific names to display. Can provide
            multiple names. Defaults to None.
        all (bool, optional): If True, show information for all cached resources of
            the specified kind. Defaults to False.

    Raises:
        typer.Abort: If kind is invalid, no names provided without --all flag,
            or if specified names are not found in cache.

    Example:
        Show a specific dataset::

            $ swm show dataset simple-pointmaze

        Show multiple datasets::

            $ swm show dataset dataset1 dataset2

        Show all cached datasets::

            $ swm show dataset --all

        Show a specific world::

            $ swm show world swm/SimplePointMaze-v0
    """
    cache_dir = data.get_cache_dir()

    if kind == "dataset":
        cached_items = data.list_datasets()
        items = names if not all else cached_items
        info_fn = data.dataset_info
        display_fn = display_dataset_info
    elif kind == "world":
        cached_items = data.list_worlds()
        items = names if not all else cached_items
        info_fn = data.world_info
        display_fn = display_world_info
    else:
        print("[red] Invalid type: must be 'world' or 'dataset' [/red]")
        raise typer.Abort()

    provided_names = list(names or [])
    if not all and not provided_names:
        print(
            "[red]Nothing to show. Use --all or provide one or more NAMES.[/red]",
        )
        raise typer.Abort()

    non_matching_local = [item for item in items if item not in cached_items]

    if len(non_matching_local) > 0:
        tree = Tree(
            f"The following {kind}s can't be found locally at `{cache_dir}`",
            style="red",
        )

        for item in non_matching_local:
            tree.add(item, style="magenta")
        print(tree)
        raise typer.Abort()

    for item in items:
        display_fn(info_fn(item))


@app.command()
def delete(
    kind: Annotated[str, typer.Argument(help="Type to delete: 'model' or 'dataset'")],
    names: Annotated[list[str], typer.Argument(help="Names of models or datasets to delete")],
):
    """Delete models or datasets from cache directory.

    Permanently removes specified models or datasets from the local cache directory.
    Requires user confirmation before deletion. This operation cannot be undone.

    Args:
        kind (str): Type of resource to delete. Must be either:
            - 'model': Delete cached world models
            - 'dataset': Delete cached datasets
        names (List[str]): One or more names of resources to delete. All specified
            names must exist in cache.

    Raises:
        typer.Abort: If kind is invalid, specified names not found in cache,
            or user cancels the confirmation prompt.

    Example:
        Delete a single dataset::

            $ swm delete dataset my-dataset

        Delete multiple models::

            $ swm delete model model1 model2

    Warning:
        This operation permanently deletes data and cannot be undone. The command
        will prompt for confirmation before proceeding.
    """
    cache_dir = data.get_cache_dir()

    if kind == "dataset":
        cached_items = data.list_datasets()
        deleter = data.delete_dataset
    elif kind == "model":
        cached_items = data.list_models()
        deleter = data.delete_model
    else:
        print("[red]Invalid type: must be 'model' or 'dataset'[/red]")
        raise typer.Abort()

    non_matching_local = [item for item in names if item not in cached_items]

    if len(non_matching_local) > 0:
        tree = Tree(
            f"The following {kind}s can't be found locally at `{cache_dir}`",
            style="red",
        )

        for item in non_matching_local:
            tree.add(item, style="magenta")
        print(tree)
        raise typer.Abort()

    typer.confirm(f"Are you sure you want to delete these cached {kind}s?", abort=True)

    for item in names:
        print(f"Deleting {kind} '{item}'...")
        deleter(item)


if __name__ == "__main__":  # pragma: no cover
    app()
