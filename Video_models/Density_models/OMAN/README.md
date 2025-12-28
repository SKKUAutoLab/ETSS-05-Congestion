# OMAN

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@article{zhu2025video,
  title={Video Individual Counting With Implicit One-to-Many Matching},
  author={Zhu, Xuhui and Xu, Jing and Wang, Bingjie and Dai, Huikang and Lu, Hao},
  journal={arXiv preprint arXiv:2506.13067},
  year={2025}
}
```

## 2. To extract the dataset, run the following script:
```shell
bash scripts/extract_dataset.sh
```

## 3. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 4. To download pretrained weights, run the following script:
```shell
bash scripts/download_weights.sh
```

## 5. To train, test and visualize the model for VSCrowd and CroHD datasets, run the following scripts:
```shell
bash scripts/train_sense.sh
bash scripts/train_crohd.sh
bash scripts/test_sense.sh
bash scripts/test_crohd.sh
bash scripts/vis_sense.sh
bash scripts/vis_crohd.sh
```

## 6. To demo the model for a video, run the following script:
```shell
bash scripts/demo.sh
```

## 7. Acknowledgement
* [tiny-smart/OMAN](https://github.com/tiny-smart/OMAN)
