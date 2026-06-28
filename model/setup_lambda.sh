#!/usr/bin/env bash
# setup_lambda.sh -- fresh Lambda A10 instance -> ready to run tmspd_probe.py
set -euo pipefail

REPO_DIR="${1:-$HOME/voyage-nexus}"   # pass your repo path as $1, default ~/voyage-nexus

# 1. tmux (session persistence across SSH drops)
sudo apt-get update -y && sudo apt-get install -y tmux

# 2. uv
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 3. venv (py3.11 for MNE) inside the repo
cd "$REPO_DIR"
uv venv --python 3.11 .venv
source .venv/bin/activate

# 4. deps -- numpy<2 pinned to avoid torch/sklearn/mne C-extension ABI mismatch
uv pip install "numpy<2" torch scipy scikit-learn mne tqdm matplotlib pandas

# 5. sanity: versions + GPU visible
python - <<'PY'
import torch, mne, sklearn, scipy, numpy as np
print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available(),
      "| numpy", np.__version__, "| mne", mne.__version__)
if not torch.cuda.is_available():
    print("!!! CUDA not visible. If the driver is older than the default wheel, "
          "reinstall: uv pip install torch --index-url https://download.pytorch.org/whl/cu121")
PY

mkdir -p logs
echo
echo "Setup done. Drop tmspd_probe.py into $REPO_DIR (next to train.py/loso.py),"
echo "stage T-MSPD so that {root}/{mode}/S{nn}/EEG/S{nn}.cdt exists, then run the probe."
