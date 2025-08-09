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

⚠️ **Note:**  before running the code, please ensure that the dataset is prepared on your device and the **corresponding paths should be added in the code** (main.py, test.py).  In this work, the training, validation, and test sets are all derived from the **CYGNSS-ERA5** dataset. The dataset contains DDMs (brcs, scatter, power), Input_N (selected auxiliary parameters), and the reference field (SWH). If you wish to incorporate additional environmental variables into the training process, please modify the corresponding parts of the code.  

2. **Independent model testing**

If you wish to test the model independently, please ensure that the paths to the corresponding **test set** and **model weight file** are added in test.py.:

```shell
python ./SCAWaveNet/test.py
```

### Project Composition (after running)

```markdown
+ SCAWaveNet
  + Saved_files
    + models
    + outputs
    + predictions
  + main.py  
  + model.py
  + test.py
  + utils.py
  + README.md
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

