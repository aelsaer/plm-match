#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a PLMLoc research environment on a fresh machine.
# This intentionally installs only code dependencies; datasets stay under DATA_ROOT.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME=${ENV_NAME:-plmloc}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
CUDA=${CUDA:-cu121}
INSTALL_TORCH=${INSTALL_TORCH:-1}
INSTALL_HLOC=${INSTALL_HLOC:-1}
INSTALL_LIGHTGLUE=${INSTALL_LIGHTGLUE:-1}

if command -v mamba >/dev/null 2>&1; then
  CONDA_CMD=mamba
elif command -v conda >/dev/null 2>&1; then
  CONDA_CMD=conda
else
  echo "Missing conda/mamba. Install Miniconda or Mambaforge first." >&2
  exit 2
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_CMD" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
conda activate "$ENV_NAME"

python -m pip install --upgrade pip wheel setuptools

if [[ "$INSTALL_TORCH" == "1" ]]; then
  case "$CUDA" in
    cpu)
      python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      ;;
    cu118|cu121|cu124)
      python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA"
      ;;
    *)
      echo "Unsupported CUDA=$CUDA. Use cpu, cu118, cu121, or cu124." >&2
      exit 2
      ;;
  esac
fi

python -m pip install -r requirements.txt
python -m pip install h5py scipy scikit-learn tqdm pandas matplotlib gdown pycolmap kornia einops faiss-cpu

mkdir -p external

if [[ "$INSTALL_HLOC" == "1" ]]; then
  if [[ ! -d external/Hierarchical-Localization/.git ]]; then
    git clone https://github.com/cvg/Hierarchical-Localization.git external/Hierarchical-Localization
  fi
  python -m pip install -e external/Hierarchical-Localization
fi

if [[ "$INSTALL_LIGHTGLUE" == "1" ]]; then
  if [[ ! -d external/LightGlue/.git ]]; then
    git clone https://github.com/cvg/LightGlue.git external/LightGlue
  fi
  python -m pip install -e external/LightGlue
fi

echo
echo "Environment ready: conda activate $ENV_NAME"
echo "For MixVPR retrieval, place the checkpoint at MixVPR/resnet50_MixVPR_large.ckpt or set MIXVPR_CKPT."
