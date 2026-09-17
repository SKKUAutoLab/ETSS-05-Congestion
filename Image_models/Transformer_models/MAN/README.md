# MAN

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{lin2022boosting,
  title={Boosting Crowd Counting via Multifaceted Attention},
  author={Lin, Hui and Ma, Zhiheng and Ji, Rongrong and Wang, Yaowei and Hong, Xiaopeng},
  booktitle={CVPR},
  year={2022}
}
```

## 2. To download weights, run the following script:
```shell
bash scripts/download_weights.sh
```

## 3. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 4. To train and test the model for ShanghaiTech, UCF-QNRF, and JHU-Crowd++ datasets, run the following scripts:
```shell
bash scripts/train_sha.sh
bash scripts/train_shb.sh
bash scripts/train_qnrf.sh
bash scripts/train_jhu.sh
bash scripts/test_sha.sh
bash scripts/test_shb.sh
bash scripts/test_qnrf.sh
bash scripts/test_jhu.sh
```

## 5. Acknowledgement
* [LoraLinH/Boosting-Crowd-Counting-via-Multifaceted-Attention](https://github.com/LoraLinH/Boosting-Crowd-Counting-via-Multifaceted-Attention)
