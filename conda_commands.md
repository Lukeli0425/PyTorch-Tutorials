# Frequent Conda Commands

Here are some useful conda commands that might help manage virtual envirnoments when building python projects.

## 1. Verify Conda installation by checking version
```zsh
conda -V
```
```zsh
conda --version
```

## 2. Show all Conda environments
```zsh
conda info -e
```
```zsh
conda env list
```

## 3. Create virtual environment
Create a virtual environment "env_name" with Python version X.X.
```zsh
conda create -n env_name python=X.X
```

## 4. Activate virtual environment
Activate an existing virtual environment "env_name".
```zsh
conda activate env_name
```
Check Python version in the environment:
```zsh
python -V
```
```zsh
python --version
```
Show all packages in the environment:
```zsh
conda list
```

## 5. Install packages in a existing environment
Install package "package_name" on virtual environment "env_name"
```zsh
conda install -n env_name package_name
```
You can also do this by activating the environment first and then install the package.
```
conda activate env_name
conda install package_name
```

## 6. Remove packages in a existing environment
Revome package "package_name" from virtual environment "env_name"
```zsh
conda uninstall -n env_name package_name
```
You can also do this by activating the environment first and then remove the package.
```
conda activate env_name
conda uninstall package_name
```

## 7. Close current virtual environment
```zsh
conda deactivate
```

## 8. Clone an existing environment
Create a new environment "env_2" by cloning a existing environment "env_1".
```zsh
conda create -n env_2 --clone env_1
```

## 9. Delete an existing environment
Delete a virtual environment "env_name".
```bash
conda remove -n env_name --all
```

## 10. Update Conda
```zsh
conda update conda
```

## 11. Add Source for Conda
```zsh
conda config --set show_channel_urls yes
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2
```

## References

[Conda Docs](https://docs.conda.io/en/latest/index.html#)
[TUNA](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/)