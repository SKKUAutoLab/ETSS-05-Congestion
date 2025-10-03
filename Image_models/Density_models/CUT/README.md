# CUT

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{cut,
title={Segmentation Assisted U-shaped Multi-scale Transformer for Crowd Counting},
author={Yifei Qian and Liangfei Zhang and Xiaopeng Hong and Carl Donovan and Ognjen Arandjelovic},
booktitle={2022 British Machine Vision Conference},
year={2022},
}
```

## 2. To download the pretrained weight, run the following script:
```shell
bash scripts/download_weight.sh
```

## 3. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 4. To train and test the model for the ShanghaiTech dataset, run the following scripts:
```shell
bash scripts/train_sha.sh
bash scripts/train_shb.sh
bash scripts/test_sha.sh
bash scripts/test_shb.sh
```

## 5. Acknowledgement
* [cha15yq/CUT](https://github.com/cha15yq/CUT)
