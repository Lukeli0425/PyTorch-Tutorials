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
conda reomve -n env_name package_name
```
You can also do this by activating the environment first and then remove the package.
```
conda activate env_name
conda remove package_name
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
```zsh
conda remove env_name --all
```

## 10. Update Conda
```zsh
conda update conda
```