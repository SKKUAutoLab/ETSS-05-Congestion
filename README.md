### Automation Lab, Sungkyunkwan University

This is the official repository of 

**OpenCounting: An Open Source Implementation of Crowd Counting Methods.**

## 1. Setup
### 1.1. Using conda
#### 1.1.1. Using environment.yml
```bash
conda env create -f envs/environment.yml
conda activate anomaly
pip install -e .
```

#### 1.1.2. Using requirements.txt
```bash
conda create --name anomaly python=3.10.13
conda activate anomaly
pip install -r envs/requirements.txt
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -e .
```

### 1.2. Using uv
```bash
uv venv anomaly --python=3.10.12
source anomaly/bin/activate
uv pip install -e .
```

## 2. Dataset Preparation
### 2.1. Image Crowd Counting Datasets
For the RGBT-CC dataset, please download it from this [link](https://lingboliu.com/RGBT_Crowd_Counting.html).

For the ShanghaiTech RGB-D dataset, please download it from this [repo](https://github.com/svip-lab/RGBD-Counting).

For the UCF-QNRF dataset, please download it from this [link](https://www.crcv.ucf.edu/data/ucf-qnrf/)

For the NWPU-Crowd dataset, please download it from this [link](https://gjy3035.github.io/NWPU-Crowd-Sample-Code/)

For the ShanghaiTech dataset, please download it from this [link](https://www.kaggle.com/datasets/tthien/shanghaitech/data)

For the STCrowd dataset, please download it from this [link](https://opendatalab.com/OpenDataLab/STCrowd)

For the NWPU-MOC dataset, please download it from this [repository](https://github.com/lyongo/NWPU-MOC)

For the Towards-vs-Away dataset, please download it from this [repository](https://github.com/rk620/Fine-Grained-CrowdCounting)

For the CARPK dataset, please download it from this [link](https://lafi.github.io/LPN/)

For the FSC147 dataset, please download it from this [link](https://www.kaggle.com/datasets/xuncngng/fsc147-0)

For the FruitNeRF dataset, please download it from this [link](https://zenodo.org/records/10869455)

For the StackCounting dataset, please download it from this [link](https://zenodo.org/records/15609540)

For the TRANCOS dataset, please download it from this [link](https://gram.web.uah.es/data/datasets/trancos/index.html)

For the PUCPR dataset, please download it from this [link](https://lafi.github.io/LPN/)

For the Crowd-SR dataset, please download it from this [repository](https://github.com/PRIS-CV/MSSRM)

For the Mall dataset, please download it from this [repository](https://github.com/fyw1999/LCSD)

For the 3DCotton dataset, please download it from this [repository](https://github.com/robotic-vision-lab/CropNeRF-A-Neural-Radiance-Field-Based-Framework)

For the TPC-268 dataset, please download it from this [repository](https://github.com/tiny-smart/TPC-268)

For the SIC dataset, please download it from this [repository](https://github.com/inbarhub/single_image_dataset)

For the DroneRGBT dataset, please download it from this [repository](https://github.com/VisDrone/DroneRGBT)

For the FSC-133 dataset, please download it from this [repository](https://github.com/ActiveVisionLab/LearningToCountAnything)

For the Penguin dataset, please download it from this [repository](https://github.com/niki-amini-naieni/CountVid)

### 2.2. Video Crowd Counting Datasets
For the FDST dataset, please download it from this [repository](https://github.com/sweetyy83/Lstn_fdst_dataset)

For the VSCrowd dataset, please download it from this [link](https://huggingface.co/datasets/HopLeeTop/VSCrowd)

For the CroHD dataset, please download it from this [link](https://motchallenge.net/data/Head_Tracking_21/)

For the CARLA dataset, please download it from this [repository](https://github.com/LeoHuang0511/FMDC)

For the MovingDroneCrowd dataset, please download it from this [repository](https://github.com/fyw1999/MovingDroneCrowd)

For the DroneBird dataset, please download it from this [repository](https://github.com/mast1ren/E-MAC)

## 3. Usage
### 3.1. Supported Models for Bayesian Crowd Counting
| Models        | UCF-QNRF           | ShanghaiTech       |
|---------------|--------------------|--------------------|
| BayesianCrowd | :heavy_check_mark: | :heavy_check_mark: |
| NoisyCC       | :heavy_check_mark: | :x:                |

### 3.2. Supported Models for Multimodal Crowd Counting
| Models  | RGBT-CC            | ShanghaiTechRGBD   | DroneRGBT          |
|---------|--------------------|--------------------|--------------------|
| CSCA    | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| IADM    | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| EAEFNet | :heavy_check_mark: | :x:                | :x:                |
| MIANet  | :heavy_check_mark: | :x:                | :x:                |
| RACANet | :x:                | :x:                | :heavy_check_mark: |
| MMFFNet | :heavy_check_mark: | :x:                | :heavy_check_mark: |

### 3.3. Supported Models for VLM Crowd Counting
| Models                    | ShanghaiTech       | NWPU-Crowd         | UCF-QNRF           | FSC-147            |
|---------------------------|--------------------|--------------------|--------------------|--------------------|
| CLIP-EBC                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| CLIP-Count                | :x:                | :x:                | :x:                | :heavy_check_mark: |
| CLIP_Counting             | :heavy_check_mark: | :x:                | :x:                | :x:                |
| LGCount                   | :x:                | :x:                | :x:                | :heavy_check_mark: |
| Count-Token-Optimization  | :x:                | :x:                | :x:                | :x:                |
| CrowdCLIP                 | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                |
| RVL_DiffGrid              | :heavy_check_mark: | :x:                | :x:                | :x:                |
| YOLO-Count                | :x:                | :x:                | :x:                | :heavy_check_mark: |
| Make-It-Count             | :x:                | :x:                | :x:                | :x:                |
| Counting-Guidance         | :x:                | :x:                | :x:                | :x:                |
| ZIP                       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| CountAnything             | :x:                | :x:                | :x:                | :x:                |

### 3.4. Supported Models for OT Crowd Counting
| Models          | ShanghaiTech       | NWPU-Crowd         | UCF-QNRF           | Arbitrary Image    |
|-----------------|--------------------|--------------------|--------------------|--------------------|
| DM-Count        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| OT-M            | :x:                | :x:                | :x:                | :heavy_check_mark: |
| GeneralizedLoss | :x:                | :x:                | :heavy_check_mark: | :x:                |

### 3.5. Supported Models for INR Crowd Counting
| Models | ShanghaiTech       | NWPU-Crowd         |
|--------|--------------------|--------------------|
| APGCC  | :heavy_check_mark: | :heavy_check_mark: |
| UNIC   | :heavy_check_mark: | :x:                |
| SI-INR | :heavy_check_mark: | :x:                |

### 3.6. Supported Models for Density Crowd Counting
| Models              | ShanghaiTech       | FDST               | UCF-QNRF           | STCrowd            | CARPK | Towards-vs-Away    | Mall               | JHU-Crowd++        | NWPU-Crowd         | TRANCOS            | Crowd-SR           | TPC-268            | VSCrowd            |
|---------------------|--------------------|--------------------|--------------------|--------------------|-------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| CSRNet              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| People-Flows        | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| S-DCNet             | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SS-DCNet            | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| GCC-SFCN            | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CACC                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SASNet              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| PAL                 | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CUT                 | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SGANet              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| RankBench           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| STCrowd             | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FGCC                | :x:                | :x:                | :x:                | :x:                | :x:   | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| P2PNet              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| UEPNet              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FIDTM               | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:   | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                |
| PML                 | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:   | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| AutoScale           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:   | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| IIM                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:   | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| DPD                 | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MSSRGN              | :x:                | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| PMLoss              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| TPC-268             | :x:                | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| STEERER             | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CCFD                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MAN                 | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:   | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| MCNN                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| mPrompt             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:   | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :heavy_check_mark: |
| KDMG                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| ResidualRegression  | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SINet               | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Fuss-Free-Structure | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |

### 3.7. Supported Models for Domain Generalization Crowd Counting
| Models     | ShanghaiTech       | UCF-QNRF           |
|------------|--------------------|--------------------|
| MPCount    | :heavy_check_mark: | :heavy_check_mark: |
| DCCUS      | :heavy_check_mark: | :x:                |
| BLA        | :heavy_check_mark: | :x:                |
| ScaleBench | :x:                | :x:                |
| DCSL       | :heavy_check_mark: | :x:                |
| SinCount   | :heavy_check_mark: | :x:                |
| MKDNet     | :heavy_check_mark: | :x:                |
| DKPNet     | :heavy_check_mark: | :x:                |
| MFFNet     | :heavy_check_mark: | :heavy_check_mark: |

### 3.8. Supported Models for Video Crowd Analysis
| Models   | SDD                | IND-TIME           | FDST               | VSCrowd            | JRDB               | HT21               | ETHUCY             |
|----------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| CrowdMAC | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

### 3.9. Supported Models for Memory Bank Crowd Counting
| Models   | JHU-Crowd++        |
|----------|--------------------|
| AWCC-Net | :heavy_check_mark: |

### 3.10. Supported Models for Transformer Crowd Counting
| Models       | JHU-Crowd++        | NWPU               | ShanghaiTech       | UCF-QNRF           | CARPK              | NWPU-MOC           | TRANCOS            | FSC-147            |
|--------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| CLTR         | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| TransCrowd   | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| PET          | :x:                | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| NWPU-MOC     | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| CountingDINO | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: |
| CCTrans      | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |

### 3.11. Supported Models for MoE Crowd Counting
| Models | ShanghaiTech       |
|--------|--------------------|
| HMoDE  | :heavy_check_mark: |

### 3.12. Supported Models for Knowledge Distillation Crowd Counting
| Models  | ShanghaiTech       | UCF-QNRF           | Mall               |
|---------|--------------------|--------------------|--------------------|
| SKT     | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| LCSD    | :x:                | :x:                | :heavy_check_mark: |
| P2RLoss | :heavy_check_mark: | :x:                | :x:                |

### 3.13. Supported Models for Domain Adaptation Crowd Counting
| Models | UCF-QNRF           | Shanghaitech       | CARPK              | PUCPR              | Arbitrary Image    |
|--------|--------------------|--------------------|--------------------|--------------------|--------------------|
| UGSDA  | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| CODA   | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| CBD    | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| DAOT   | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| CDFA   | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: |

### 3.14. Supported Models for GCN Crowd Counting
| Models     | UCF-QNRF           | JHU-Crowd++        | ShanghaiTech       |
|------------|--------------------|--------------------|--------------------|
| Gramformer | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| GAAL       | :x:                | :x:                | :heavy_check_mark: |
| MDGCN      | :x:                | :x:                | :x:                |
| DSGCNet    | :x:                | :x:                | :heavy_check_mark: |

### 3.15. Supported Models for GCN Video Crowd Counting
| Models | FDST               |
|--------|--------------------|
| STGN   | :heavy_check_mark: |

### 3.16. Supported Models for Open-world Object Counting
| Models | ShanghaiTech       | FSC147             |
|--------|--------------------|--------------------|
| OVID   | :heavy_check_mark: | :heavy_check_mark: |

### 3.17. Supported Models for Density Video Crowd Counting
| Models           | VSCrowd            | CroHD              | FDST               | CARLA              | MovingDroneCrowd   | DroneBird          |
|------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| VSCrowd          | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| DAANet           | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                |
| AVCC             | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| OMAN             | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| FMDC             | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                |
| CGNet            | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| DRNet            | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| MovingDroneCrowd | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| DroneBird        | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: |
| CLRNet           | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CrowdGAN         | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |

### 3.18. Supported Models for Low-shot Crowd Counting
| Models | FSC147             |
|--------|--------------------|
| FamNet | :heavy_check_mark: |

### 3.19. Supported Models for 3D Crowd Counting
| Models      | FruitNeRF          | StackCounting      | 3DCotton           |
|-------------|--------------------|--------------------|--------------------|
| FruitNeRF   | :heavy_check_mark: | :x:                | :x:                |
| FruitNeRF++ | :heavy_check_mark: | :x:                | :x:                |
| 3DC         | :x:                | :heavy_check_mark: | :x:                |
| CropNeRF    | :x:                | :x:                | :heavy_check_mark: |

### 3.20. Supported Models for Active Learning Crowd Counting
| Models   | SIC                |
|----------|--------------------|
| Count_AL | :heavy_check_mark: |

### 3.21. Supported Models for Segment Anything Crowd Counting
| Models         | FSC-147            | COCO2017           | Arbitrary Image    | ShanghaiTech       | CARPK              | Penguin            |
|----------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| CountAnything  | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| Count-Anything | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| AdaSEEMCount   | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| SAM3Count      | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

### 3.22. Supported Models for Weakly-supervised Crowd Counting
| Models                  | ShanghaiTech       | FSC-133            |
|-------------------------|--------------------|--------------------|
| CCRanking               | :heavy_check_mark: | :x:                |
| LearningToCountAnything | :x:                | :heavy_check_mark: |

### 3.23. Supported Models for Semi-supervised Crowd Counting
| Models | ShanghaiTech       |
|--------|--------------------|
| DREAM  | :heavy_check_mark: |
| TMTB   | :heavy_check_mark: |

### 3.24. Supported Models for Diffusion Crowd Counting
| Models           | ShanghaiTech       |
|------------------|--------------------|
| CrowdDiff        | :heavy_check_mark: |
| ControlNet-Crowd | :heavy_check_mark: |

## 4. Citation
If you find our work useful, please cite the following:
```
@misc{Chi2025,
  author       = {Chi Tran},
  title        = {OpenCounting: An Open Source Implementation of Crowd Counting Methods},
  publisher    = {GitHub},
  booktitle    = {GitHub repository},
  howpublished = {https://github.com/IceIce1ce/OpenCounting},
  year         = {2025}
}
```

## 5. Contact
If you have any questions, feel free to contact `Chi Tran` 
([ctran743@gmail.com](ctran743@gmail.com) or [tdc2000@skku.edu](tdc2000@skku.edu)).

## 6. Acknowledgement
Our framework is built using multiple open source, thanks for their great contributions.