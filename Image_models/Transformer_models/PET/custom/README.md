# PET

![arch](assets/arch.JPG)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{liu2023point,
  title={Point-query quadtree for crowd counting, localization, and more},
  author={Liu, Chengxin and Lu, Hao and Cao, Zhiguo and Liu, Tongliang},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={1676--1685},
  year={2023}
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

## 4. To train and test the model for ShanghaiTech and CARPK datasets, run the following scripts:
```shell
bash scripts/train_sha.sh
bash scripts/train_carpk.sh
bash scripts/test_sha.sh
bash scripts/test_carpk.sh
```

## 5. Acknowledgement
* [tiny-smart/remote-sensing-object-counting-and-localization](https://github.com/tiny-smart/remote-sensing-object-counting-and-localization)
