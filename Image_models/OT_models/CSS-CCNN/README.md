# CSS-CCNN

![arch](assets/arch.jpg)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@article{CSSCNN20,
    title = {Completely Self-Supervised Crowd Counting via Distribution Matching},
    author = {Babu Sam, Deepak and Agarwalla, Abhinav and Joseph, Jimmy and Sindagi, A. Vishwanath and Babu, R. Venkatesh and Patel, M. Vishal},
    journal = {arXiv preprint arXiv:2009.06420},
    Year = {2020}
}
```

## 2. To download the weight, run the following script:
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
bash scripts/test_sha.sh
```

## 5. Acknowledgement
* [val-iisc/css-ccnn](https://github.com/val-iisc/css-ccnn)
