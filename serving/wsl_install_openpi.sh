#!/bin/bash
# Install uv, clone openpi and create its environment (WSL). Log: eval_results/openpi_install.log
set -e
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version
cd ~
if [ ! -d openpi ]; then
  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
fi
cd openpi
git log --oneline -1
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
.venv/bin/python -c "import jax, openpi; print('jax', jax.__version__, jax.devices())"
echo INSTALL_OK
