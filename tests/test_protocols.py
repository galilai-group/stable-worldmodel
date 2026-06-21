"""Protocol import-contract tests (PR1 G8).

Verifies that Costable has a single canonical definition in protocols.py and that
all historical import paths remain resolvable after the dedup.
"""


def test_costable_protocol_has_single_canonical_identity():
    """solver.solver.Costable must be the same object as protocols.Costable."""
    from stable_worldmodel.protocols import Costable as PublicCostable
    from stable_worldmodel.solver.solver import Costable as SolverCostable

    assert PublicCostable is SolverCostable


def test_existing_protocol_import_paths_remain_available():
    """All previously importable protocols must still resolve after the dedup."""
    from stable_worldmodel.protocols import Actionable, Costable, Transformable
    from stable_worldmodel.solver.solver import Solver

    assert Actionable is not None
    assert Costable is not None
    assert Transformable is not None
    assert Solver is not None
