import stable_worldmodel as swm


world = swm.World(
    "swm/PushT-v1",
    num_envs=5,
    image_shape=(224, 224),
    render_mode="rgb_array",
)

print("Available variations: ", world.single_variation_space.names())

world.set_policy(swm.policy.RandomPolicy())
world.record_dataset(
    "example-pusht",
    episodes=10,
    seed=2347,
    options=None,
)

swm.pretraining(
    "scripts/train/dinowm.py",
    dataset_name="example-pusht",
    output_model_name="dummy_pusht",
    dump_object=True,
)

model = swm.policy.AutoCostModel("dummy_pusht").to("cuda")
config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
solver = swm.solver.CEMSolver(model, num_samples=300, var_scale=1.0, n_steps=30, topk=30, device="cuda")
policy = swm.policy.WorldModelPolicy(solver=solver, config=config)

world.set_policy(policy)
results = world.evaluate(episodes=5, seed=2347)

print(results)
