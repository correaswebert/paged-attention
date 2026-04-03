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

Architecture
------------

Let's start basic with a single GPU executor with no parallelism. The first task
is to implement paged attention. After a working prototype, the other features
can be added.

References:
* https://docs.vllm.ai/en/latest/design/paged_attention
* https://docs.vllm.ai/en/latest/design/arch_overview
* https://docs.vllm.ai/en/latest/design/model_runner_v2


### Scheduler

* submit_request - place the new request in the waitq
* execute_request - move request from waitq to runq and run forward

### KV Cache Manager

* allocate - issue a new block
* free - free the issued block
* append - add a new token to the end of the block
* copy - copy entire block to a newly allocated block
* fork - increase the refcount of the block


TODO
----

- [ ] Scheduler
- [ ] Cache manager
- [ ] Executor
- [ ] Engine
- [ ] Processor
