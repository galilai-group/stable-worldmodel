"""stable-worldmodel CLI commands."""

from typing import Any, Dict, List, Optional, Union

import typer
from rich import print
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.console import Group
from typing_extensions import Annotated
from rich.rule import Rule

from stable_worldmodel import data

from .__about__ import __version__

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel

console = Console()


def _summarize(x: Any) -> str:
    try:
        a = np.asarray(x)
    except:
        return repr(x)
    return (
        f"shape={list(a.shape)}, min={a.min()}, max={a.max()}"
        if a.size
        else f"[] shape={list(a.shape)}"
    )


def _leaf(m: Any) -> bool:
    return isinstance(m, dict) and "type" in m


def _leaf_table(title: str, m: Dict[str, Any]) -> Table:
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
    root: Dict[str, Dict] = {}
    if not flat_names:
        return root
    for raw in flat_names:
        parts = [p for p in str(raw).split(sep) if p]
        cur = root
        for p in parts:
            cur = cur.setdefault(p, {})
    return root


def _render_hierarchy(parent: Tree, d: Dict[str, Dict]):
    for k in sorted(d):
        is_non_leaf = bool(d[k])
        label = Text(k, style="bold cyan") if is_non_leaf else Text(k)
        child = parent.add(label) if parent else Tree(label)
        if is_non_leaf:
            _render_hierarchy(child, d[k])


def _render(meta: Union[Dict[str, Any], List[Any], None], label: str):
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


def _variation_space(variation: Dict[str, Any], title: str = "Variation Space"):
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
        if isinstance(names, (list, tuple)) and names:
            tree_dict = _build_hierarchy(names, sep=".")
            _render_hierarchy(vroot, tree_dict)
        else:
            vroot.add(Text("names: —", style="dim"))

    return vroot


def display_world_info(info: Dict[str, Any]) -> None:
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


def display_dataset_info(info: Dict[str, Any]) -> None:
    """Pretty-print dataset info as a Rich table with a dotted separator."""
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
    """Show installed stable-worldmodel version."""
    if value:
        typer.echo(f"stable-worldmodel version: {__version__}")
        raise typer.Exit()


@app.callback()
def common(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            callback=_version_callback,
            help="Show installed stable-wordlnodel version.",
        ),
    ] = None,
):
    """Common options for all commands."""
    pass


@app.command("list")
def list_cmd(
    kind: Annotated[
        str, typer.Argument(help="Type to list: 'model', 'dataset' or 'world'")
    ],
):
    """List stable-worldmodel models/datasets/worlds stored in cache dir."""
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
    names: Annotated[
        Optional[List[str]], typer.Argument(help="Names of worlds or datasets to show")
    ] = None,
    all: Annotated[
        bool,
        typer.Option(
            "--all", "-a", help="Show all cached datasets/worlds", is_flag=True
        ),
    ] = False,
):
    """Show information about cached datasets or worlds."""
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
    names: Annotated[
        List[str], typer.Argument(help="Names of models or datasets to delete")
    ],
):
    """Delete models or datasets from cached dir."""
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

    # _show_dataset_table(datasets_to_delete, "Delete local Minari datasets")
    typer.confirm(f"Are you sure you want to delete these cached {kind}s?", abort=True)

    for item in names:
        print(f"Deleting {kind} '{item}'...")
        deleter(item)


if __name__ == "__main__":
    app()
