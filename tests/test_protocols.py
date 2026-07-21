"""Protocol import-contract tests (PR1 G8).

Verifies that Costable has a single canonical definition in protocols.py and that
all historical import paths remain resolvable after the dedup.
"""


def test_costable_protocol_has_single_canonical_identity():
    """solver.solver.Costable must be the same object as protocols.Costable."""
    from stable_worldmodel.protocols import Costable as PublicCostable
    from stable_worldmodel.planning.solver.solver import (
        Costable as SolverCostable,
    )

    assert PublicCostable is SolverCostable


def test_existing_protocol_import_paths_remain_available():
    """All previously importable protocols must still resolve after the dedup."""
    from stable_worldmodel.protocols import Actionable, Costable, Transformable
    from stable_worldmodel.planning.solver.solver import Solver

    assert Actionable is not None
    assert Costable is not None
    assert Transformable is not None
    assert Solver is not None


def test_constrainable_protocol_has_single_canonical_identity():
    """solver.solver.Constrainable must be the same object as protocols.Constrainable."""
    from stable_worldmodel.protocols import (
        Constrainable as PublicConstrainable,
    )
    from stable_worldmodel.planning.solver.solver import (
        Constrainable as SolverConstrainable,
    )

    assert PublicConstrainable is SolverConstrainable


def test_planning_protocols_are_re_exported_from_protocols():
    """Dynamics/Objective live in protocols.py; planning re-exports the same objects."""
    from stable_worldmodel import planning
    from stable_worldmodel.protocols import (
        Constrainable,
        Costable,
        Dynamics,
        Objective,
    )

    assert planning.Dynamics is Dynamics
    assert planning.Objective is Objective
    assert planning.Costable is Costable
    assert planning.Constrainable is Constrainable


def test_cost_evaluator_constrainable_detection():
    """ShootingCostEvaluator satisfies Constrainable iff it was built with constraints."""
    import torch

    from stable_worldmodel.planning import (
        ControlPenalty,
        ShootingCostEvaluator,
        GoalMSE,
    )
    from stable_worldmodel.protocols import Constrainable

    class _Dyn(torch.nn.Module):
        def encode(self, x):
            return x

        def rollout(self, info, ac):
            info['predicted_emb'] = ac
            return info

    plain = ShootingCostEvaluator(_Dyn(), GoalMSE())
    constrained = ShootingCostEvaluator(
        _Dyn(), GoalMSE(), constraints=[ControlPenalty()]
    )

    assert not isinstance(plain, Constrainable)
    assert isinstance(constrained, Constrainable)
