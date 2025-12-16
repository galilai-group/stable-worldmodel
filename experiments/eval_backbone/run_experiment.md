# run

```
python experiments/eval_backbone/run.py policy=pyro_test_epoch_20 world.max_episode_steps=200
```

## Backbone eval
```
python experiments/eval_backbone/run.py policy=resnet50_epoch_10,vit_mae_base_epoch_10,vit_base_epoch_10,siglip2_base16_224_epoch_10,dinov2_small_epoch_10,dino_vits16_epoch_10,dinov3_vits16_epoch_10 world.max_episode_steps=200
```

## Encoder scaling
```
python experiments/eval_backbone/run.py -m policy=epoch_20/dinov2_small_epoch_20,epoch_20/pusht_dinov2_base_psmall_epoch_20,epoch_20/pusht_dinov2_large_psmall_epoch_20,epoch_20/pusht_dinov2_giant_psmall_epoch seed=42,3072,9261 launcher=lucas_eval
```

## Predictor scaling
```
python experiments/eval_backbone/run.py -m policy=epoch_20/pusht_dinov2_small_ptiny_epoch_20,epoch_20/pusht_dinov2_small_psmall_epoch_20,epoch_20/pusht_dinov2_small_pbase_epoch_20,epoch_20/pusht_dinov2_small_plarge_epoch_20 seed=42,3072,9261 launcher=lucas_eval
```
