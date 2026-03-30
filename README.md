Paged Attention
===============

This is an implementation of paged-attention in python.

Build
-----

First let us set up the virtual environment. We'll use `uv` for faster
installation and its simple drop-in usage.

```sh
uv venv --python 3.12
source .venv/bin/activate
```

Now let us install the requirements.

```sh
uv pip install numpy torch typer

cd src/cuda
uv pip install -e . --no-build-isolation
```

TODO
----

- [ ] Scheduler
- [ ] Cache manager
- [ ] Executor
- [ ] Engine
- [ ] Processor
