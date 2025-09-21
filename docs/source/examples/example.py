import stable_worldmodel as swm
import torch

# create world
world = swm.World(
    "swm/SimplePointMaze-v0",
    num_envs=7,
    image_shape=(224, 224),
    render_mode="rgb_array",
)

# collect data for pre-training
world.set_policy(swm.policy.RandomPolicy())
world.policy.set_seed(42)
world.record_dataset("simple-pointmaze", episodes=10, seed=2347)
world.record_video("./", seed=2347)

# pre-train world model
swm.pretraining("scripts/train/dummy.py", "++dump_object=True dataset_name=simple-pointmaze output_model_name=dummy_test")

# evaluate world model
# world.set_policy(swm.policy.AutoPolicy("output_model_name"))
action_dim = world.envs.single_action_space.shape[0]
cost_fn = torch.nn.functional.mse_loss
world_model = swm.wm.DummyWorldModel((224, 224, 3), action_dim)
solver = swm.solver.RandomSolver(horizon=5, action_dim=action_dim, cost_fn=cost_fn)
policy = swm.policy.WorldModelPolicy(world_model, solver, horizon=10, action_block=5, receding_horizon=5)
world.set_policy(policy)

spt_module = torch.load(swm.utils.get_cache_dir()+"/dummy_test_object.ckpt", weights_only=False)
world_model = spt_module.model
results = world.evaluate(episodes=2, seed=2347)#, options={...})

# what about eval on all type of env?
# TODO: add leaderboard

print(results)