# natvar

## Installation and setup

Clone the repository with 

```bash
git clone https://github.com/AddisonHowe/natvar.git
cd natvar
```
and create a conda environment by running
```bash
conda env create -p ./env -f environment.yml
conda activate env
python -m pip install -e ".[dev,jax,jupyter]"
```

Check that tests pass:
```bash
pytest tests
```
