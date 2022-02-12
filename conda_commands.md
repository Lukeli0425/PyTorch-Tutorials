# Frequent Conda Commands

Here are some useful conda commands that might help manage virtual envirnoments when running python projects.

## 1. Verify Installation by checking version
```zsh
conda -V
```
```zsh
conda --version
```
My output is:
```
conda 4.11.0
```

## 2. Show all environments
```zsh
conda info -e
```
```zsh
conda env list
```
## 3. Create virtual environment

Create a virtual environment with name "env_name" and Python version X.X.

```zsh
conda create -n env_name python=X.X
```
## 4. Activate virtual environment

Activate an existing virtual environment "env_name".

```zsh
conda activate env_name
```

## 5.Install packages in a existing envornment

```zsh
conda install -n env_name package_name
```

You can also do this by activating the environment first and install the package.
```
conda activate env_name
conda install package_name
```