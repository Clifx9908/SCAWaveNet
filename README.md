# SCAWaveNet

**A Spatial-Channel Attention-Based Network for Global Significant Wave Height Retrieval**

<div align=center>
    <img src="./pics/SCAWaveNet_architecture.jpg" alt="SCAWaveNet_architecture"/>
</div> 


## Abstract

Recent advancements in spaceborne GNSS missions have produced extensive global datasets, providing a robust basis for deep learning-based significant wave height (SWH) retrieval. While existing deep learning models predominantly utilize CYGNSS data with four-channel information, they often adopt single-channel inputs or simple channel concatenation without leveraging the benefits of cross-channel information interaction during training. To address this limitation, a novel spatial–channel attention-based network, namely SCAWaveNet, is proposed for SWH retrieval. Specifically, features from each channel of the DDMs are modeled as independent attention heads, enabling the fusion of spatial and channel-wise information. For auxiliary parameters, a lightweight attention mechanism is designed to assign weights along the spatial and channel dimensions. The final feature integrates both spatial and channel-level characteristics. Model performance is evaluated using four-channel CYGNSS data. When ERA5 is used as a reference, SCAWaveNet achieves an average RMSE of 0.438 m. When using buoy data from NDBC, the average RMSE reaches 0.432 m. Compared to state-of-the-art models, SCAWaveNet reduces the average RMSE by at least 3.52% on the ERA5 dataset and by 5.68% on the NDBC buoy observations.

## Getting Started

### Environment Setup:

1. **Python >=3.8, torch>=2.0.0**

⚠️ **Note:** before installing torch, please check the **CUDA version** and refer to the official  [PyTorch website](https://pytorch.org/get-started/previous-versions/) to find the appropriate installation command. 

For example:

```shell
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

2. **Installation of other dependencies**

```shell
pip install -r requirements.txt
```

⚠️ **Note:**  if the code fails to run due to the NumPy version, it may be necessary to downgrade NumPy to a version below 2.0

3. **Recommended GPU Memory**

GPU Memory should be more than **8 GB**.

### Code Running:

1. **Overall code running**

```shell
python ./SCAWaveNet/main.py
```

⚠️ **Note:**  before running the code, please ensure that the dataset is prepared on your device and the **corresponding paths should be added in the code** (main.py).  In this work, the training, validation, and test sets are all derived from the **CYGNSS-ERA5** dataset. The dataset contains DDMs (brcs, scatter, power), Input_N (selected auxiliary parameters), and the reference field (SWH). If you wish to incorporate additional environmental variables into the training process, please modify the corresponding parts of the code.  

2. **Independent model testing**

If you wish to test the model independently, please ensure that the paths to the corresponding **test set** and **model weight file** are added in test.py:

```shell
python ./SCAWaveNet/test.py
```

### Project Composition (after running)

```asciiarmor
SCAWaveNet
├── Saved_files
│   ├── models
│   │   └── model_best_rmse_exp{x}.pth
│   ├── outputs
│   │   └── valid_results_epoch{x}_exp{x}.pdf
│   └── predictions
│       └── model_best_rmse_exp{x}_{xxx}_predictions.mat
├── main.py  
├── model.py
├── test.py
├── utils.py
└── README.md
```

## Citation

If our work is helpful to you, please cite it as follows:

```
@article{zhang2025scawavenet,
  title={SCAWaveNet: A Spatial-Channel Attention-based Network for Global Significant Wave Height Retrieval},
  author={Zhang, Chong and Liu, Xichao and Zhan, Yibing and Tao, Dapeng and Ni, Jun},
  journal={arXiv preprint arXiv:2507.00701},
  year={2025}
}
```



## ToDo

- [x] Release SCAWaveNet training code.
- [x] Release SCAWaveNet evaluation and metrics code.
- [ ] Release other code in the project.
