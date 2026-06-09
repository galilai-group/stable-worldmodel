"""Declarative ManiSkill3 task registry.

This list is the single extension point for ManiSkill envs. Each entry maps a
``swm/...`` id to a ManiSkill ``task_id``, with optional per-task overrides
(``robot_uids``, ``control_mode``, ``camera_name``, ...). Everything except
``id`` is forwarded to ``ManiSkillWrapper`` as kwargs.

Adding a robot or task is a one-line entry here — no wrapper code changes:

* New task, default robot:
    ``dict(id='swm/MS<Name>-v0', task_id='<ManiSkillTask-v1>')``
* Existing task on a different embodiment (ManiSkill swaps it; the wrapper
  picks up the new action_space automatically):
    ``dict(id='swm/...-v0', task_id='PickCube-v1', robot_uids='widowx')``
* Robot-specific task variant: point ``task_id`` at it (e.g. ``'PickCubeSO100-v1'``).
* Uniform action across arms: add ``control_mode='pd_ee_delta_pose'``.
"""

TASK_SPECS = [
    # --- Franka Panda table-top manipulation (Panda is the task default) ---
    dict(id='swm/MSPickCube-v0', task_id='PickCube-v1'),
    dict(id='swm/MSPushCube-v0', task_id='PushCube-v1'),
    dict(id='swm/MSPullCube-v0', task_id='PullCube-v1'),
    dict(id='swm/MSPokeCube-v0', task_id='PokeCube-v1'),
    dict(id='swm/MSStackCube-v0', task_id='StackCube-v1'),
    dict(id='swm/MSLiftPegUpright-v0', task_id='LiftPegUpright-v1'),
    dict(id='swm/MSPegInsertionSide-v0', task_id='PegInsertionSide-v1'),
    dict(id='swm/MSPlugCharger-v0', task_id='PlugCharger-v1'),
    dict(id='swm/MSPickSingleYCB-v0', task_id='PickSingleYCB-v1'),
    dict(id='swm/MSRollBall-v0', task_id='RollBall-v1'),
    # --- SIMPLER / real2sim Bridge digital twins (WidowX is the task default) ---
    dict(
        id='swm/SimplerCarrotOnPlate-v0', task_id='PutCarrotOnPlateInScene-v1'
    ),
    dict(
        id='swm/SimplerSpoonOnTowel-v0',
        task_id='PutSpoonOnTableClothInScene-v1',
    ),
    dict(
        id='swm/SimplerStackCube-v0',
        task_id='StackGreenCubeOnYellowCubeBakedTexInScene-v1',
    ),
    dict(
        id='swm/SimplerEggplantInBasket-v0',
        task_id='PutEggplantInBasketScene-v1',
    ),
    # --- Examples of extending to other robots (uncomment / copy a line) ---
    # dict(id='swm/MSPickCubeWidowX-v0', task_id='PickCube-v1', robot_uids='widowx'),
    # dict(id='swm/MSPickCubeSO100-v0', task_id='PickCubeSO100-v1'),
    # dict(id='swm/MSPickCubeEE-v0', task_id='PickCube-v1', control_mode='pd_ee_delta_pose'),
]
