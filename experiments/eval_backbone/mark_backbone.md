# 1. Download the ckpt

```bash
gdown --id 1hW6Z8NB87Wv3z_2Q5Ad6UWgivJTPBpyk -O /tmp/archive.tar.gz \
&& tar -xzf /tmp/archive.tar.gz -C "$STABLEWM_HOME"
```

# 2. Eval them

```bash
python experiments/eval_backbone/run.py --config-name=pusht.yaml -m policy=0b91b34e-fa2c-463d-bc01-e8eae6061e71/pusht_dinov2_large_psmall_epoch_20,10ddcb77-6b00-46f3-982f-78d6224a11e5/pusht_dino_vits16_psmall_epoch_20,1484654d-4f58-4dcc-be87-5a440245fc75/pusht_vit_base_psmall_epoch_20,1b15a68a-15e7-4844-b552-cc99561867ec/pusht_dinov2_base_psmall_epoch_20,1e2513b8-b970-4e6b-ab35-639d7d405920/pusht_dinov2_small_psmall_epoch_20,207e97ad-012a-4e03-aec8-74006769d9ca/pusht_dinov3_vits16_psmall_epoch_20,25cc8d6e-6745-4082-89ad-777c101899db/pusht_dinov2_small_psmall_epoch_20,271a36d0-4e1f-4b6b-bd79-7c9e592939dc/pusht_dinov2_giant_psmall_epoch_20,4070ba27-5fdb-45e4-b8f8-6a84d479f5c1/pusht_dinov2_small_psmall_epoch_20,41dc047b-514e-42e1-af3a-797f04966031/pusht_dinov2_small_psmall_epoch_20,4eecf203-a7ac-4385-a535-331753d9b8e6/pusht_vjepa2_large_psmall_epoch_20,537693f4-10b8-4f2d-b8fc-64990d6b083c/pusht_dinov2_small_psmall_epoch_20,67b6a31d-337f-4400-8dba-c699978eb8bd/pusht_dinov2_small_psmall_epoch_20,7739ab02-d6b4-4eff-b911-531db9d62d4c/pusht_dinov2_small_psmall_epoch_20,7a39ccff-09fd-4989-8875-cf56a50e7d46/pusht_dinov2_small_psmall_epoch_20,8774db0b-a4d1-4421-af09-14cefedf90c5/pusht_dinov2_base_psmall_epoch_20,8dfb96b4-5c90-43dd-8b0b-6626c9d58130/pusht_resnet50_psmall_epoch_20,8e2b73b9-b8b5-4ef2-bbfa-a572224044f5/pusht_dinov2_small_psmall_epoch_20,8e44e114-c8de-49b9-ad8f-312454f7ca97/pusht_dinov2_small_psmall_epoch_20,9c829a3c-991e-4f6a-864c-f1901325ad24/pusht_vit_mae_base_psmall_epoch_20,a7d5acdc-1599-4ffa-b1a1-27d34fed48b4/pusht_dinov2_small_psmall_epoch_20,a7d8a2a7-3ccc-4278-925d-c3636e4e630d/pusht_siglip2_base16_224_psmall_epoch_20,ad635c1b-eb1a-4634-8578-04f71bdc4fee/pusht_dinov2_giant_psmall_epoch_20,b4326e28-01e0-4b09-b249-df6097fa0759/pusht_dinov2_small_psmall_epoch_20,b93414f5-4c9b-4417-8362-247e30896f86/pusht_dinov2_small_psmall_epoch_20,cab74b64-fcd4-4da4-9301-a0e664f31433/pusht_dinov2_small_psmall_epoch_20,d31d98e9-9322-4d85-a9bd-f9193efff07d/pusht_dinov2_large_psmall_epoch_20,d6df6809-4723-4e88-ad8a-07a55010ad44/pusht_ijepa_vith14_22k_psmall_epoch_20,fbc94710-2837-4048-a7d6-b672f6566db9/pusht_dinov2_small_psmall_epoch_20 seed=42,3072,9261 launcher=YOUR_LAUNCHER
```
