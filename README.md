# PyTorch-Tutorials

## About

This repo includes resources, codes and notes while learning [**PyTorch**](https://pytorch.org/).

## Environment Setup

All my codes are built on an M1 Macbook Air(Apple Silicon). Therefore we need to install  [**Miniforge**](https://github.com/conda-forge/miniforge) from [here](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh) since Anaconda and Miniconda don't work on Apple Silicon. 

After installation, we can create an virtual environment via Miniforge:
```zsh
conda create --n pytorch python=3.8
```
I named it "pytorch" since I'm only using it when learning PyTorch. Note that we need to create a py38 environment if we want to run PyTorch on Apple Silicon. Acitivate the environment:
```zsh
conda activate pytorch
```
Now we can install PyTorch on our environment:
```zsh
conda install pytorch torchvision -c pytorch
```
Verify the installation by importing torch & torchvision and printing their versions:
```python
import torch
import torchvision
print(f"torch=={torch.__version__}")
print(f"torchvision=={torchvision.__version__}")
print("Hello PyTorch!")
```
If no errors were raised, the installation is successful. My output is:
```
torch==1.10.2
torchvision==0.11.0a0
Hello PyTorch!
```
Here we install some other packages in case of future usage:
```zsh
conda install matplotlib scikit-learn pandas
conda install jupyter notebook
```

## File Listings

## References

[PyTorch Docs](https://pytorch.org/)

[Tutorial Videos on Bilibili](https://www.bilibili.com/video/BV1US4y1M7fg?p=1)