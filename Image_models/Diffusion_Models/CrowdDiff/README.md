# CrowdDiff

![arch](assets/arch.jpg)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{ranasinghe2024crowddiff,
  title={CrowdDiff: Multi-hypothesis crowd density estimation using diffusion models},
  author={Ranasinghe, Yasiru and Nair, Nithin Gopalakrishnan and Bandara, Wele Gedara Chaminda and Patel, Vishal M},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={12809--12819},
  year={2024}
}
```

## 2. To install the environment, run the following script:
```shell
bash scripts/install.sh
```

## 3. To download the pretrained weight, run the following script:
```shell
bash scripts/download_weight.sh
```

## 4. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 5. To train and test the model for the ShanghaiTech dataset, run the following scripts:
```shell
bash scripts/train_sha.sh
bash scripts/test_sha.sh
```

## 6. Acknowledgement
* [dylran/crowddiff](https://github.com/dylran/crowddiff)
