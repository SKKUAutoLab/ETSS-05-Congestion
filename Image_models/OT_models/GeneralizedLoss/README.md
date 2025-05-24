# GeneralizedLoss

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@InProceedings{Wan_2021_CVPR,
    author    = {Wan, Jia and Liu, Ziquan and Chan, Antoni B.},
    title     = {A Generalized Loss Function for Crowd Counting and Localization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2021},
    pages     = {1974-1983}
}
```

## 2. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 3. To train and test the model for the UCF-QNRF dataset, run the following scripts:
```shell
bash scripts/train_qnrf.sh
bash scripts/test_qnrf.sh
```

## 4. Acknowledgement
* [jia-wan/GeneralizedLoss-Counting-Pytorch](https://github.com/jia-wan/GeneralizedLoss-Counting-Pytorch)