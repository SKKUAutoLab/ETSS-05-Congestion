# CGNet

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{liu2024weakly,
  title={Weakly Supervised Video Individual Counting},
  author={Liu, Xinyan and Li, Guorong and Qi, Yuankai and Yan, Ziheng and Han, Zhenjun and van den Hengel, Anton and Yang, Ming-Hsuan and Huang, Qingming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19228--19237},
  year={2024}
}
```

## 2. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 3. To train, test, and visualize the model for VSCrowd and CroHD datasets, run the following scripts:
```shell
bash scripts/train_sense.sh
bash scripts/train_crohd.sh
bash scripts/test_sense.sh
bash scripts/test_crohd.sh
bash scripts/vis_sense.sh
bash scripts/vis_crohd.sh
```

## 4. Acknowledgement
* [streamer-AP/CGNet](https://github.com/streamer-AP/CGNet)
