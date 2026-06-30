"""
V0.5 Wire Harness — 5 movers, ONE target configuration per episode.

Fully learnable rework of v0_4_1 (no deterministic fallback, no constraint
override). One SAC model is trained per target configuration (stage 0..4);
at execution time the per-stage models are chained in any order.

Cable topology (star, hub = platform1), rest lengths measured from XML:
    platform1 ↔ Wire1 ↔ platform2   (1.315 m)
    platform1 ↔ Wire2 ↔ platform3   (3.006 m)
    platform1 ↔ Wire3 ↔ platform6   (2.697 m)
    platform1 ↔ Wire4 ↔ platform7   (2.400 m)
"""

import math
import os

MOVER_STARTS = [
    [4.0, 3.0],   # platform1 (RED)   — hub
    [5.0, 2.0],   # platform2 (GREEN) — Wire1
    [1.0, 2.3],   # platform3 (YELLOW)— Wire2
    [1.5, 1.8],   # platform6 (PURPLE)— Wire3
    [4.0, 0.5],   # platform7 (ORANGE)— Wire4
]

MOVER_BODY_NAMES  = ["platform1", "platform2", "platform3", "platform6", "platform7"]
MOVER_JOINT_NAMES = ["slide_joint1", "slide_joint2", "slide_joint3", "slide_joint6", "slide_joint7"]
VEL = 2.0

# 5 target configurations (configuration-major: row k = Konfiguration k,
# entries in mover order: platform1 RED, platform2 GREEN, platform3 YELLOW,
# platform6 PURPLE, platform7 ORANGE). Stage-k SAC learns row k.
# All p1↔p2 distances ≤ 1.5 m (Wire1 constraint verified).
MOVER_TARGETS = [
    # Konfiguration 0: Rechts oben Cluster
    [[5.2, 2.8], [5.3, 1.6], [2.5, 2.8], [3.1, 1.5], [5.1, 0.5]],

    # Konfiguration 1: Rechts unten Cluster
    [[5.2, 0.6], [4.0, 0.3], [5.4, 3.4], [5.0, 3.0], [2.8, 1.1]],

    # Konfiguration 2: Mitte Cluster
    [[3.0, 2.8], [3.5, 1.6], [0.5, 2.8], [1.1, 1.5], [3.1, 0.4]],

    # Konfiguration 3: Links Cluster
    [[3.2, 0.4], [2.0, 0.3], [3.4, 3.4], [3.0, 3.0], [0.8, 0.7]],

    # Konfiguration 4: Finale Positionen (Taping-Stationen)
    [[1.2, 0.8], [1.0, 2.0], [4.2, 0.4], [3.6, 0.8], [1.4, 3.2]],
]

N_STAGES = len(MOVER_TARGETS)

# ── Episode ────────────────────────────────────────────────────────────────────
SIMEND      = 60     # 60 × 60 = 3 600 max steps — one configuration per episode
GOAL_RADIUS = 0.15

# ── Observation / reward ──────────────────────────────────────────────────────
# Pairwise offsets are normalized by cable rest length (cabled pairs) or the
# field diagonal (uncabled pairs); ~1.0 then means "cable fully stretched".
CABLE_PAIR_LENGTHS = {
    (0, 1): 1.315,   # Wire1
    (0, 2): 3.006,   # Wire2
    (0, 3): 2.697,   # Wire3
    (0, 4): 2.400,   # Wire4
}
FIELD_DIAG = math.sqrt(6.72**2 + 3.84**2)

W_OBSTACLE_MAP = 0.1     # per cell of the 5×5 local obstacle map (ludwig_sb3)
W_CABLE_MAP    = 0.005   # per cell of the 7×7 local cable map   (ludwig_sb3)

# Which cables (1=Wire1..4=Wire4) are attached to each mover (config_base reference).
# A mover's OWN cables are filtered out of its local maps — its trailing cable is
# unavoidable and would otherwise add constant penalty/observation noise.
CABLE_CONNECT = [
    [1, 2, 3, 4],   # RED (hub) — all 4 cables
    [1],            # GREEN  — Wire1
    [2],            # YELLOW — Wire2
    [3],            # PURPLE — Wire3
    [4],            # ORANGE — Wire4
]

# Cable body IDs adjacent to each platform attachment (config_base reference).
# Excluded from the global collision map so attachment stubs don't register as
# permanent collisions right under the platforms (matches environment_ludwig_sb3).
CABLE_START_MU = [
    [1, 2, 3, 38, 39, 40, 68, 69, 70, 71, 72, 73],  # RED
    [8, 9, 10],                                      # GREEN
    [11, 12, 13],                                    # YELLOW
    [41, 42, 43],                                    # PURPLE
    [88, 89, 90],                                    # ORANGE
]

# ── SB3 SAC (per stage) ───────────────────────────────────────────────────────
TOTAL_STEPS    = 3_000_000   # env steps per stage. With gradient_steps=-1 this is
                             # also the gradient-step budget. SAC is far more
                             # sample-efficient than A2C — 0.5–1M is often enough;
                             # lower this once a stage converges.
N_ENVS         = 12          # SubprocVecEnv workers feeding the shared replay buffer
                             # (sized for ~5 concurrent stage runs on a 64-core box)
EVAL_FREQ      = 50_000
N_EVAL_EPS     = 3
CKPT_FREQ      = 100_000
LOG_DIR        = "logs/sac"          # learn script appends /stage_{k}
CKPT_DIR       = "checkpoints/sac"   # learn script appends /stage_{k}

POLICY_NET_ARCH = [256, 256]  # roomy for the 30-dim obs (2*N + N*(N-1))

# Off-policy SAC: continuous actions in [-1, 1], automatic entropy temperature.
SAC_LR                     = 3e-4
SAC_BUFFER_SIZE            = 1_000_000
SAC_LEARNING_STARTS        = 10_000   # random actions to seed the buffer first
SAC_BATCH_SIZE             = 256
SAC_TAU                    = 0.005    # target-network Polyak factor
SAC_GAMMA                  = 0.99
SAC_TRAIN_FREQ             = 1        # collect one vec-step (N_ENVS transitions) per rollout
SAC_GRADIENT_STEPS         = -1       # then do as many grad steps as transitions collected (1:1 with N_ENVS)
SAC_ENT_COEF               = "auto"   # learned entropy temperature
SAC_TARGET_UPDATE_INTERVAL = 1

# ── XML ───────────────────────────────────────────────────────────────────────
XML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "WireHarness012e.xml"
)
 