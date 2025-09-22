# Part code of Uni4D
## Official implement of Uni4D by PyTorch
## Device: 2 × RTX 4090

## Installation
The code is tested with Python 3.10, PyTorch 2.0.1, torchvision 0.15.2, and CUDA 11.8.

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413) and Chamfer_Distance_Loss:
```
cd modules
python setup.py install

cd ./extensions/chamfer_dist
python setup.py install
```

## Related Repositories  
We thank the authors of related repositories:
1. PSTNet: https://github.com/hehefan/Point-Spatio-Temporal-Convolution
2. P4Transformer: https://github.com/hehefan/P4Transformer
3. MAE: https://github.com/facebookresearch/mae

