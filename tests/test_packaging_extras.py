"""Per-environment extras must cover their environment's imports.

Regression test for #343: the `env` extra is all-or-nothing and pulls
simulators whose native builds fail on stock pip images (labmaze and
box2d-py via gymnasium[all]), so pip aborted and a user needing a single
environment could not install anything. The per-environment extras
(`env-tworoom`, `env-pusht`) let one environment be installed alone;
these tests pin them to what those environments actually import.
"""

import ast
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

pytestmark = pytest.mark.skipif(
    tomllib is None, reason='tomllib requires Python 3.11+'
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Import name -> providing distribution, for names where they differ.
# Unlisted third-party imports are assumed to share the import name.
IMPORT_ALIASES = {
    'cv2': 'opencv-python-headless',
}

# Extra -> env-module trees whose third-party imports it must cover.
EXTRA_TREES = {
    'env-tworoom': ['stable_worldmodel/envs/two_room'],
    'env-pusht': ['stable_worldmodel/envs/pusht'],
}

# Shared envs modules these trees import relatively (from ..utils import
# ...), invisible to a per-tree scan.
SHARED_MODULES = ['stable_worldmodel/envs/utils.py']


def _req_name(requirement):
    """Lowercase distribution name of a PEP 508 requirement string."""
    name = requirement.split(';')[0]
    for sep in ('[', '==', '>', '<', '!', '~', ' '):
        name = name.split(sep)[0]
    return name.strip().lower()


def _extras():
    data = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text())
    return data['project']['optional-dependencies']


def _third_party_imports(paths):
    roots = set()
    skip = set(sys.stdlib_module_names) | {'stable_worldmodel'}
    for raw in paths:
        path = REPO_ROOT / raw
        files = sorted(path.rglob('*.py')) if path.is_dir() else [path]
        for file in files:
            tree = ast.parse(file.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots.update(
                        alias.name.split('.')[0] for alias in node.names
                    )
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module
                ):
                    roots.add(node.module.split('.')[0])
    return roots - skip


def test_per_env_extra_covers_its_env_imports():
    extras = _extras()
    base = {_req_name(req) for req in _base_requirements()}
    for extra, trees in EXTRA_TREES.items():
        available = base | {_req_name(req) for req in extras[extra]}
        imports = _third_party_imports(trees + SHARED_MODULES)
        missing = {}
        for module in sorted(imports):
            dist = IMPORT_ALIASES.get(module, module)
            if dist not in available:
                missing[module] = dist
        assert not missing, (
            f'{extra} is missing requirements for imports of '
            f'{trees}: {missing}'
        )


def _base_requirements():
    with open(REPO_ROOT / 'pyproject.toml', 'rb') as f:
        project = tomllib.load(f)['project']
    return project['dependencies']


def test_full_env_extra_still_covers_per_env_extras():
    extras = _extras()
    env = {_req_name(req) for req in extras['env']}
    for extra in EXTRA_TREES:
        missing = set(map(_req_name, extras[extra])) - env
        assert not missing, (
            f'{extra} requirements missing from the full env extra: {missing}'
        )
