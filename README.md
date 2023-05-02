<div align="center">

# USA Net

A robust method for mapping an environment and doing language-based planning.

</div>

<br />

## Getting Started

Install from [pypi](https://pypi.org/project/usa-net/):

```bash
pip install usa-net
```

Install from Github:

```bash
pip install git+https://github.com/codekansas/usa.git
```

Alternatively, clone and install:

```bash
git clone git@github.com:codekansas/usa.git
cd usa && pip install -e .
```

There are some extra dependencies for the notebooks which can be installed with:

```bash
pip install 'usa-net[ipynb]'
```

See the [notebooks](/notebooks) for examples.

### Requirements

- Python 3.10+
- PyTorch 2.0+

### Development

Add pre-commit hooks to clean Jupyter notebooks:

```bash
pre-commit install
```
