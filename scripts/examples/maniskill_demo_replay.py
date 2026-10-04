"""Demo-replay success-rate check for the swm ManiSkill envs.

Replays ManiSkill's official demonstration trajectories through the swm
``ManiSkillWrapper`` and reports the success rate. The demos are successful by
construction, so a correct env + ``success -> terminated`` wiring should
reproduce a high success rate — giving a real number without a trained policy.

Reproduction uses the demo's recorded **initial env_state** (not the seed):
batched-GPU demos are not seed-reproducible in a single env, but restoring the
exact initial state and replaying the recorded actions is deterministic.

Requires a CUDA + Vulkan GPU and ``stable-worldmodel[maniskill]``.

Usage:
    # Download demos once, then replay:
    python -m mani_skill.utils.download_demo PickCube-v1
    python scripts/examples/maniskill_demo_replay.py \
        --swm-id swm/MSPickCube-v0 --task PickCube-v1 --episodes 20
"""

import argparse
import glob
import json
import os

import gymnasium as gym
import h5py

import stable_worldmodel  # noqa: F401  (registers swm/* envs)


def _find_demo(task_id):
    """Locate a downloaded demo (.h5/.json) for the task.

    Prefer motionplanning/teleop demos: they use absolute joint-position control
    (``pd_joint_pos``), which replays deterministically from a restored initial
    state. RL demos use relative end-effector deltas (``pd_ee_delta_pos``) that
    drift under open-loop replay, so they reproduce only partially.
    """
    root = os.path.expanduser(f'~/.maniskill/demos/{task_id}')
    jsons = sorted(
        glob.glob(os.path.join(root, '**', '*.json'), recursive=True)
    )
    if not jsons:
        raise FileNotFoundError(
            f'No demos under {root}. Run: '
            f'python -m mani_skill.utils.download_demo {task_id}'
        )
    order = {'motionplanning': 0, 'teleop': 1, 'rl': 2}

    def rank(p):
        for name, r in order.items():
            if f'/{name}/' in p:
                return r
        return 3

    jsons.sort(key=lambda p: (rank(p), p))
    return jsons[0]


def _state_at(group, idx):
    """Reconstruct ManiSkill set_state_dict input from the h5 env_states."""
    import torch

    state = {'actors': {}, 'articulations': {}}
    for kind in ('actors', 'articulations'):
        grp = group.get(f'env_states/{kind}')
        if grp is None:
            continue
        for name in grp:
            row = grp[name][idx]
            state[kind][name] = torch.as_tensor(row[None])
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--swm-id', default='swm/MSPickCube-v0')
    ap.add_argument('--task', default='PickCube-v1')
    ap.add_argument('--episodes', type=int, default=20)
    args = ap.parse_args()

    meta_path = _find_demo(args.task)
    h5_path = meta_path.replace('.json', '.h5')
    meta = json.load(open(meta_path))
    episodes = meta['episodes']
    control_mode = meta['env_info']['env_kwargs'].get('control_mode')
    print(f'demo: {meta_path}\ncontrol_mode: {control_mode}')

    env = gym.make(args.swm_id, control_mode=control_mode)
    raw = env.unwrapped
    f = h5py.File(h5_path, 'r')

    n = min(args.episodes, len(episodes))
    successes = 0
    for i in range(n):
        ep = episodes[i]
        g = f[f'traj_{ep["episode_id"]}']
        actions = g['actions'][:]
        env.reset(seed=ep.get('episode_seed', 0))
        raw.set_state_dict(_state_at(g, 0))  # exact initial state
        ever = False
        for a in actions:
            _, _, term, _, info = env.step(a)
            ever = ever or bool(info.get('success'))
        successes += int(ever)
        print(f'  ep{i}: success={ever}')

    rate = 100.0 * successes / n
    print(f'\nsuccess_rate: {successes}/{n} = {rate:.1f}%')
    env.close()


if __name__ == '__main__':
    main()
