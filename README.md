# PyTorch-Tutorials

## About

This repo includes resources, codes and notes while learning [**PyTorch**](https://pytorch.org/).

## Environment Setup

All my codes are built on an M1 Macbook Air(Apple Silicon). Therefore we need to install  [Miniforge](https://github.com/conda-forge/miniforge) from [here](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh) since Anaconda and Miniconda don't work on Apple Silicon. 

After installation, we can create an virtual environment via Miniforge:
```zsh
conda create --n pytorch python=3.8
```
I named it "pytorch" since I'm only using it when learning PyTorch. Note that we need to create a py38 environment if we want to run PyTorch on Apple Silicon. Now we can install PyTorch on our environment:
```zsh
conda install pytorch torchvision -c pytorch
```
Verify the installation by importing torch adn torchvision and printing the versions:
```python
import torch
import torchvision
print(torch.__version__)
```

## File Listings


