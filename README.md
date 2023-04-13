# USA Net

USA net project code.

## Getting Started

Install from [pypi](https://pypi.org/project/usa-net/):

```bash
pip install usa-net
```

### Requirements

- Python 3.10+
- PyTorch 2.0+

### Development

Add this to your `.git/config` file to automatically clear the output of notebooks before committing:

```ini
[filter "strip-notebook-output"]
  clean = "jupyter nbconvert --clear-output --to notebook --stdin --stdout --log-level=ERROR"
```
