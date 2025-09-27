"""Minari CLI commands."""

import os
from collections import deque
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
from typing_extensions import Annotated

import typer
from gymnasium.envs.registration import EnvSpec
from rich import print
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from __about__ import __version__

import stable_worldmodel as swm

from datasets import load_dataset_builder


app = typer.Typer()


def _version_callback(value: bool):
    """Show installed stable-worldmodel version."""
    if value:
        typer.echo(f"stable-worldmodel version: {__version__}")
        raise typer.Exit()


class TableTree:
    def __init__(
        self,
        name: Optional[str] = None,
        total_episodes: int = 0,
        total_steps: int = 0,
        size: float = 0,
        authors: Optional[set] = None,
        count: int = 0,
        sub_nodes: Optional[Dict] = None,
        docs_url: Optional[str] = None,
    ):
        self.name = name
        self.total_episodes = total_episodes
        self.total_steps = total_steps
        self.size = size
        self.authors = authors if authors is not None else set()
        self.count = count
        self.sub_nodes = sub_nodes if sub_nodes is not None else {}
        self.docs_url = docs_url

    def update(self, other):
        self.total_episodes += other.total_episodes
        self.total_steps += other.total_steps
        self.size += other.size
        self.authors.update(other.authors)
        self.count += other.count

    def to_row(self) -> List[str]:
        if len(self.authors) == 0:
            authors = "Unknown"
        else:
            authors = ", ".join(self.authors)

        name = self.name
        assert name is not None
        if len(self.sub_nodes) > 0:
            name = f"[bold]{name}/...[/bold]\n [italic]Group with {self.count} datasets[/italic]"
        if self.docs_url is not None:
            name = f"[link={self.docs_url}]{name}[/link]"

        return [
            name,
            TableTree.print_num(self.total_episodes),
            TableTree.print_num(self.total_steps),
            self.print_size(),
            authors,
        ]

    def print_size(self) -> str:
        if self.size < 1_000:
            return f"{self.size:.1f} MB"
        elif self.size < 1_000_000:
            return f"{self.size / 1_000:.2f} GB"
        else:
            return f"{self.size / 1_000_000:.2f} TB"

    @staticmethod
    def print_num(num: float) -> str:
        num_letters = ["", "K", "M", "B", "T"]
        i = 0
        while num >= 1_000 and i < len(num_letters):
            num /= 1000
            i += 1

        return f"{num:.0f}{num_letters[i]}"


def _show_dataset_table(datasets: Dict[str, Dict[str, Any]], table_title: str):
    MAX_ROWS_PER_GROUP = 10

    table_tree = TableTree()
    for dataset_id in datasets.keys():
        dataset_metadata = datasets[dataset_id]
        dataset_node = TableTree(
            name=dataset_id,
            total_episodes=dataset_metadata["total_episodes"],
            total_steps=dataset_metadata["total_steps"],
            size=dataset_metadata["dataset_size"],
            authors=dataset_metadata.get("author"),
            count=1,
            docs_url=dataset_metadata.get("docs_url"),
        )

        table_tree.update(dataset_node)
        namespace, _, _ = parse_dataset_id(dataset_id)
        current_root = table_tree
        for ns in namespace_hierarchy(namespace):
            if ns not in current_root.sub_nodes:
                current_root.sub_nodes[ns] = TableTree(name=ns)
            current_root = current_root.sub_nodes[ns]
            current_root.update(dataset_node)
        current_root.sub_nodes[dataset_id] = dataset_node

    caption = (
        f"Number of datasets: {table_tree.count}, total size: {table_tree.print_size()}"
    )
    table = Table(title=table_title, caption=caption)
    table.add_column("Name", justify="left", style="cyan", no_wrap=True)
    table.add_column("# Episodes", justify="right", style="green", no_wrap=True)
    table.add_column("# Steps", justify="right", style="green", no_wrap=True)
    table.add_column("Size", justify="left", style="green", no_wrap=True)
    table.add_column("Author", justify="left", style="magenta", no_wrap=True)

    queue = deque(table_tree.sub_nodes.values())
    section_sentinel = object()
    aggregate = False
    while queue:
        table_node = queue.popleft()
        if table_node is section_sentinel:
            table.add_section()
            continue

        if len(table_node.sub_nodes) == 0:
            table.add_row(*table_node.to_row())
        elif aggregate and len(table_node.sub_nodes) > MAX_ROWS_PER_GROUP:
            table.add_row(*table_node.to_row())
            table.add_section()
        else:
            queue.extend(table_node.sub_nodes.values())
            queue.append(section_sentinel)
        aggregate = aggregate or len(table_node.sub_nodes) > 1

    print(table)


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
    all: Annotated[
        bool, typer.Option("--all", "-a", help="Show all dataset versions.")
    ] = False,
    prefix: Annotated[
        Optional[str], typer.Option("--prefix", "-p", help="Filter datasets by prefix.")
    ] = None,
):
    """List stable-worldmodel datasets stored in local directory."""
    datasets = local.list_local_datasets(
        latest_version=True, compatible_minari_version=not all, prefix=prefix
    )

    dataset_dir = swm.utils.get_cache_dir()
    table_title = f"Local saved datasets('{dataset_dir}')"

    _show_dataset_table(datasets, table_title)


@app.command()
def show(dataset_id: Annotated[str, typer.Argument()]):
    """Describe a local or remote dataset, and its environment."""
    local_datasets = local.list_local_datasets(prefix=dataset_id)
    if dataset_id in local_datasets:
        dst_metadata = local_datasets[dataset_id]
    else:
        remote_path = None
        if "://" in dataset_id:
            remote_type, remote_path = dataset_id.split("://", maxsplit=1)
            remote_path, dataset_id = remote_path.split("/", maxsplit=1)
            remote_path = f"{remote_type}://{remote_path}"
        remote_datasets = hosting.list_remote_datasets(
            remote_path=remote_path, prefix=dataset_id
        )

        if dataset_id in remote_datasets:
            dst_metadata = remote_datasets[dataset_id]
        else:
            local_dataset_path = get_dataset_path()
            print(
                Text(
                    f"""The dataset `{dataset_id}` can't be found locally """
                    f"""(at `{local_dataset_path}`) or remotely.""",
                    style="red",
                )
            )
            raise typer.Abort()

    description = dst_metadata.get("description")
    docs_url = dst_metadata.get("docs_url")

    if docs_url is not None:
        dataset_id_text = f"[{dataset_id}]({docs_url})"
    else:
        dataset_id_text = dataset_id

    dataset_spec_table = Table(show_header=False, highlight=True)
    dataset_spec_table.add_column(style="bold")
    dataset_spec_table.add_column(style="not bold")

    for key, value in get_dataset_spec_dict(dst_metadata).items():
        md = Markdown(
            str(value), inline_code_lexer="python", inline_code_theme="monokai"
        )
        dataset_spec_table.add_row(key, md)

    print(Markdown(f"""# {dataset_id_text}"""))

    if description is not None:
        print(Markdown(f"""\n## Description\n {description} """))

    print(Markdown("## Dataset Specs"))
    print(dataset_spec_table)

    for env_type in ["env_spec", "eval_env_spec"]:
        env_spec_json = dst_metadata.get(env_type)

        if env_spec_json is not None:
            assert isinstance(env_spec_json, str)
            env_spec_json = (  # for gymnasium 1.0.0 compatibility
                env_spec_json.replace('"order_enforce": true,', "")
                .replace('"apply_api_compatibility": false,', "")
                .replace('"autoreset": false, ', "")
            )
            env_spec = EnvSpec.from_json(env_spec_json)
            env_spec_table = Table(show_header=False, highlight=True)
            env_spec_table.add_column(style="bold")
            env_spec_table.add_column(style="not bold")

            for key, value in get_env_spec_dict(env_spec).items():
                md = Markdown(
                    value, inline_code_lexer="python", inline_code_theme="monokai"
                )
                env_spec_table.add_row(key, md)

            if env_type == "env_spec":
                print(Markdown("## Environment Specs"))
            else:
                print(Markdown("## Evaluation Environment Specs"))

            print(env_spec_table)


@app.command()
def delete(datasets: Annotated[List[str], typer.Argument()]):
    """Delete datasets from local database."""
    # check that the given local datasets exist
    local_dsts = local.list_local_datasets()
    non_matching_local = [dst for dst in datasets if dst not in local_dsts]
    if len(non_matching_local) > 0:
        local_dataset_path = get_dataset_path()
        tree = Tree(
            f"The following datasets can't be found locally at `{local_dataset_path}`",
            style="red",
        )
        for dst in non_matching_local:
            tree.add(dst, style="magenta")
        print(tree)
        raise typer.Abort()

    # prompt to delete datasets
    datasets_to_delete = {
        local_name: local_dsts[local_name]
        for local_name in local_dsts.keys()
        if local_name in datasets
    }

    _show_dataset_table(datasets_to_delete, "Delete local Minari datasets")
    typer.confirm("Are you sure you want to delete these local datasets?", abort=True)

    cache_dir = swm.utils.get_cache_dir()
    for dst in datasets:
        try:
            dataset_path = Path(cache_dir, dst)
            # remove cache files
            builder = load_dataset_builder(
                "parquet", data_files=str(Path(dataset_path, "*.parquet"))
            )
            builder.cleanup_cache_files()

            # delete dataset directory
            shutil.rmtree(dataset_path, ignore_errors=True)
            print(f"Deleted dataset {dst}!")

        except Exception as e:
            print(f"[red]Error cleaning up dataset [cyan]{dst}[/cyan]: {e}[/red]")


if __name__ == "__main__":
    app()
