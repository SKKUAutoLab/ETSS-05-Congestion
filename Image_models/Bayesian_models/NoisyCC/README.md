# NoisyCC

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@article{wan2020modeling,
  title={Modeling Noisy Annotations for Crowd Counting},
  author={Wan, Jia and Chan, Antoni},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
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
* [jia-wan/NoisyCC-pytorch](https://github.com/jia-wan/NoisyCC-pytorch)
