# AI teaches me AI

This is my logbook from my interactive lessons where ChatGPT teaches me ML

The logs from each day are located in year/month/day directories. The directories have 'a' prefix, because Python does not run directories that are just numbers for its modules.

## Python setup

See [docs/python_setup.md](docs/python_setup.md).


## VSCode search regexp

Finds markdown blocks in .py file exported from a Jupyter notebook.

```regexp
^#\s*%%\s*\[markdown\][\s\S]*?\n\n
```
