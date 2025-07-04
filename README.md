## Learning Refined Representation for Unsupervised Visible-Infrared Person Re-Identification with Hierarchical Identity

## Data Preparation Steps

1. Place the SYSU-MM01 and RegDB datasets in the following directories:
   - `./data/sysu`
   - `./data/regdb`

2. Run the scripts to prepare the training data (convert to Market-1501 format):
   - `python prepare_sysu.py`
   - `python prepare_regdb.py`

## Model Preparation Steps

We adopt the self-supervised pre-trained models (ViT-B/16+ICS) from [Self-Supervised Pre-Training for Transformer-Based Person Re-Identification](https://github.com/damo-cv/TransReID-SSL?tab=readme-ov-file).
**Download link:** https://drive.google.com/file/d/1ZFMCBZ-lNFMeBD5K8PtJYJfYEk5D9isd/view

Before standard training, we train the pre-trained models for the initial 30 epochs from[Shallow-Deep Collaborative Learning for Unsupervised Visible-Infrared  Person Re-Identification](https://github.com/yangbincv/SDCL). 
**Download link:** https://pan.baidu.com/s/1hkdtDOPuQOy0z54J-pyZXg?pwd=qgvv

## Training and Evaluation Steps

We utilize 2 A100 GPUs for training.

**SYSU-MM01:**

Train:

```shell
sh train_cc_vit_sysu.sh
```

Test:

```shell
sh test_cc_vit_sysu.sh
```

**RegDB:**

Train:

```shell
sh train_cc_vit_regdb.sh
```

Test:

```shell
sh test_cc_vit_regdb.sh
```

## Training Examples
**Download link:** https://pan.baidu.com/s/1Ao8hxJC9FTFfBraMinXbCQ?pwd=v7yh
