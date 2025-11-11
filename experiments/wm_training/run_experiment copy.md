# How to run these experiments?

### 1. clone the repo

```bash
git clone https://github.com/rbalestr-lab/stable-worldmodel
```

### 2. create a conda env + install the library

```bash
conda create --name swm python=3.10
conda activate
cd stable-worldmodel/
pip install -e stable-worlmodel/
```

### (optional) change your cache dir

Every checkpoints and dataset will be saved in a cache directory. By default the path is `~/.stable-worldmodel/`. If you wish to change this, open your `~/.bashrc` file and add the following line:

```bash
export STABLEWM_HOME="/path/to/dir"
```

### download pusht dataset and original model checkpoint

Install google drive downloader package: `pip install gdown`

Then, execute:

```bash
gdowm --id 1f-3y5kWVKwtqH5YGNPhDP-BzSQnomZnV
gdowm --id 1je3nRgGERd-U_4dHyNcwMpJW66w8_YBQ
tar --use-compress-program=unzstd -xvf dataset_train.tar.zst -C /path/to/stable-worldmodel-cachedir
tar --use-compress-program=unzstd -xvf dataset_val.tar.zst -C /path/to/stable-worldmodel-cachedir
```

### You are now ready to queue experiments

launch these command

```bash
# ResNet models
# python experiments/wm_training/run.py backbone=microsoft/resnet-18 output_model_name=resnet18
python experiments/wm_training/run.py backbone=microsoft/resnet-50 output_model_name=resnet50

# ViT models
python experiments/wm_training/run.py backbone=google/vit-base-patch16-224 output_model_name=vit_base

# DINO v1
# python experiments/wm_training/run.py backbone=facebook/dino-vitb16 output_model_name=dino_vitb16
python experiments/wm_training/run.py backbone=facebook/dino-vits16 output_model_name=dino_vits16

# DINOv2
python experiments/wm_training/run.py backbone=facebook/dinov2-small output_model_name=dinov2_small

# DINOv3
python experiments/wm_training/run.py backbone=facebook/dinov3-vits16-pretrain-lvd1689m output_model_name=dinov3_vits16
#python experiments/wm_training/run.py backbone=facebook/dinov3-vitb16-pretrain-lvd1689m output_model_name=dinov3_vitb16

# MAE
python experiments/wm_training/run.py backbone=facebook/vit-mae-base output_model_name=vit_mae_base

# IJEPA
#python experiments/wm_training/run.py backbone=facebook/ijepa_vith14_1k output_model_name=ijepa_vith14_1k
python experiments/wm_training/run.py backbone=facebook/ijepa_vith14_22k output_model_name=ijepa_vith14_22k

# CLIP
python experiments/wm_training/run.py backbone=timm/vit_base_patch32_clip_224.metaclip_400m output_model_name=metaclip_vit_base
```
