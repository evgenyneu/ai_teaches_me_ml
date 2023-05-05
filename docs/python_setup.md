# Python code set up

Here is how to install specific versions of Python and its libraries for this project.

## Install Python

1. [Install asdf](https://asdf-vm.com/#/core-manage-asdf), which allows to run different versions on Python for different local projects, instead of using one system-wide Python installation. It is equivalent of rbenv in Ruby or NVM in Node.

2. Add Python plugin to asdf:

```
asdf plugin add python
```

3. Install Python version specified in [.tool-versions](/.tool-versions).

```
asdf install
```

4. If you followed [asdf installation manual](https://asdf-vm.com/#/core-manage-asdf) from step #1 and added `. $HOME/.asdf/asdf.sh` to your shell (`.bashrc`, `.bash_profile ` etc.) then correct version of Python will automatically be used (specified in [.tool-versions](/.tool-versions)) when you change into the project's directory directory.


## Create Python environment

Make sure you are in this project root direcotry and run:

```
python -m venv .venv
```

This creates `.venv` directory for storing Python and its libraries. This way, the Python libraries stay **local** to this project and do not interfere with other projects on your computer. It uses [venv](https://docs.python.org/3/library/venv.html) module that is now built into Python. Venv is equivalent to Bundler in Ruby or to NPM in Node. There is no need to use external tools for managing Python environments anymore, like Conda or virtualenv.

## Activate Python environment

```
. .venv/bin/activate
```

Note: better yet, [setup auto-activation](https://stackoverflow.com/a/50830617/297131).


## Install Python libraries

Project's dependencies are specified in [requirements.in](/requirements.in). To
read it, install pip-tools:

```
pip install pip-tools
```

Generate dependencies in [requirements.txt](/requirements.txt):

```
pip-compile requirements.in
```

Your application should always work with the dependencies installed from this generated requirements.txt. If you have to update a dependency you just need to update the requirements.in file and redo pip-compile.

### Install dependencies

This will install dependencies from `requirements.txt` in local `.venv` directory.

```
pip-sync
```

## Deactivate Python environment (optional)

When finished working with this project:

```
deactivate
```

This will deactivate your local Python environment, so you can use your global system-wide Python and its libraries if needed.
