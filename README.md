# RBF_identification


## Installation 
This repository depends on another one (for LMI resolution). Do not forget to init git submodules
```bash
$ git submodule update --init
```

If you want to create virtual environment, you can use the following command:
```bash
$ python -m venv env
$ source env/bin/activate
```

Then you can install the required packages using the following command:
```bash
pip install -r requirements.txt
```

## Setup

To use this repository, you need to update the PYTHONPATH variable. It will add the path to the LMI-toolbox package as well as the rbf model package. Run this command:
```bash
source ./setup.sh
```

## Virtual Environment with Jupyter

First, make sure the virtual environment is activated.   
To use the virtual environment as a jupyter kernel, you need to do the following command:
```bash
pip install ipykernel
python -m ipykernel install --user --name=env
```

Check this [website](https://janakiev.com/blog/jupyter-virtual-envs/) for more details.
