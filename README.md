# SCAWaveNet
A Spatial-Channel Attention-based Network for Global Significant Wave Height Retrieval

<div align=center>
    <img src="./pics/SCAWaveNet_architecture.jpg" alt="SCAWaveNet_architecture"/>
</div> 

# Abstract
Recent advancements in spaceborne GNSS missions have produced extensive global datasets, providing a robust basis for deep learning-based significant wave height (SWH) retrieval. While existing deep learning models predominantly utilize CYGNSS data with four-channel information, they often adopt single-channel inputs or simple channel concatenation without leveraging the benefits of cross-channel information interaction during training. To address this limitation, a novel spatial–channel attention-based network, namely SCAWaveNet, is proposed for SWH retrieval. Specifically, features from each channel of the DDMs are modeled as independent attention heads, enabling the fusion of spatial and channel-wise information. For auxiliary parameters, a lightweight attention mechanism is designed to assign weights along the spatial and channel dimensions. The final feature integrates both spatial and channel-level characteristics. Model performance is evaluated using four-channel CYGNSS data. When ERA5 is used as a reference, SCAWaveNet achieves an average RMSE of 0.438 m. When using buoy data from NDBC, the average RMSE reaches 0.432 m. Compared to state-of-the-art models, SCAWaveNet reduces the average RMSE by at least 3.52% on the ERA5 dataset and by 5.68% on the NDBC buoy observations.

# Getting Started

```
pip install -r requirements.txt
```

# Citation

If our work is helpful to you, please cite it as follows:

```
@article{zhang2025scawavenet,
  title={SCAWaveNet: A Spatial-Channel Attention-based Network for Global Significant Wave Height Retrieval},
  author={Zhang, Chong and Liu, Xichao and Zhan, Yibing and Tao, Dapeng and Ni, Jun},
  journal={arXiv preprint arXiv:2507.00701},
  year={2025}
}
```

# ToDo

- [x] Release SCAWaveNet training code
- [x] Release SCAWaveNet evaluation and metrics code
- [ ] Release other code
