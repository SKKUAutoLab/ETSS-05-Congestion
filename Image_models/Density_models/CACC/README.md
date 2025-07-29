# CACC

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@InProceedings{Liu_2019_CVPR,
author = {Liu, Weizhe and Salzmann, Mathieu and Fua, Pascal},
title = {Context-Aware Crowd Counting},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## 2. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 3. To train and test the model for the ShanghaiTech dataset, run the following scripts:
```shell
bash scripts/train_sha.sh
bash scripts/train_shb.sh
bash scripts/test_sha.sh
bash scripts/test_shb.sh
```

## 4. Acknowledgement
* [weizheliu/Context-Aware-Crowd-Counting](https://github.com/weizheliu/Context-Aware-Crowd-Counting)
