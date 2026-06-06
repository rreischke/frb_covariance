This code implements the covariance induced by the large-scale structure (LSS) of the observed Dispersion Measure (DM) for a list of FRBs with observed redshift and position on the sky.

## Documentation, Installation and Examples

### 1) Clone the repository

```shell
git clone git@github.com:rreischke/frb_covariance.git
cd frb_covariance
```

### 2) Create a clean Python environment

`cosmopower` depends on TensorFlow and is most reliable in a dedicated environment.
Following the official `cosmopower` recommendation, use Python 3.11.

```shell
conda create -n frb_cov_env python=3.11 pip
conda activate frb_cov_env
```

### 3) Install `cosmopower` first

Install from PyPI:

```shell
pip install cosmopower
```

Alternative (conda-forge):

```shell
conda install -c conda-forge cosmopower
```

Quick check:

```shell
python -c "import cosmopower as cp; print(cp.__version__)"
```

If you do not have a GPU, TensorFlow may print a GPU warning. This is expected and can be ignored.

### 4) Install this package

```shell
pip install -e .
```

### 5) Verify imports

```shell
python -c "import frb_cov; import cosmopower; print('ok')"
```

### Notes

- The emulators used by the examples are expected in the local `cosmopower/` directory in this repository.
- A small run example is available in `feedback_fits/test_covariance.ipynb`.

If you use the code, please cite [the paper](https://ui.adsabs.harvard.edu/abs/2023MNRAS.524.2237R/abstract).

