# TransCrowd

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@article{liang2022transcrowd,
  title={TransCrowd: weakly-supervised crowd counting with transformers},
  author={Liang, Dingkang and Chen, Xiwu and Xu, Wei and Zhou, Yu and Bai, Xiang},
  journal={Science China Information Sciences},
  volume={65},
  number={6},
  pages={1--14},
  year={2022},
  publisher={Springer}
}
```

## 2. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 3. To download the pretrained weight, run the following script:
```shell
bash scripts/download_weight.sh
```

## 4. To train and test the model for ShanghaiTech and QNRF datasets, run the following scripts:
```shell
bash scripts/train_sha.sh
bash scripts/train_shb.sh
bash scripts/train_qnrf.sh
bash scripts/test_sha.sh
bash scripts/test_shb.sh
bash scripts/test_qnrf.sh
```

## 5. Acknowledgement
* [dk-liang/TransCrowd](https://github.com/dk-liang/TransCrowd)
