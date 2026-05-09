#!/usr/bin/env bash

set -euo pipefail

ACCELERATOR="${IMF_ACCELERATOR:-auto}"
JAX_CUDA_VARIANT="${JAX_CUDA_VARIANT:-cuda13}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
FORCE_TORCH_INSTALL="${IMF_FORCE_TORCH_INSTALL:-0}"

if [[ "$ACCELERATOR" == "auto" ]]; then
	if command -v nvidia-smi >/dev/null 2>&1; then
		ACCELERATOR="gpu"
	else
		ACCELERATOR="cpu"
	fi
fi

echo "Installing iMeanFlow dependencies for accelerator: $ACCELERATOR"
python3 -m pip install --upgrade pip

case "$ACCELERATOR" in
	tpu)
		python3 -m pip install "jax[tpu]==0.4.27" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
		python3 -m pip install "jaxlib==0.4.27" "flax>=0.8"
		;;
	gpu)
		# On modern NVIDIA systems like RTX 5090, prefer the bundled CUDA wheels.
		python3 -m pip install --upgrade "jax[$JAX_CUDA_VARIANT]" "flax>=0.8"
		if [[ "$FORCE_TORCH_INSTALL" == "1" ]]; then
			python3 -m pip install --upgrade torch torchvision --index-url "$PYTORCH_INDEX_URL"
		elif ! python3 - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") else 1)
PY
		then
			python3 -m pip install --upgrade torch torchvision --index-url "$PYTORCH_INDEX_URL"
		else
			echo "torch already installed; leaving existing torch/torchvision untouched"
		fi
		;;
	cpu)
		python3 -m pip install --upgrade jax "flax>=0.8"
		if [[ "$FORCE_TORCH_INSTALL" == "1" ]]; then
			python3 -m pip install --upgrade torch torchvision
		elif ! python3 - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") else 1)
PY
		then
			python3 -m pip install --upgrade torch torchvision
		else
			echo "torch already installed; leaving existing torch/torchvision untouched"
		fi
		;;
	*)
		echo "Unsupported IMF_ACCELERATOR value: $ACCELERATOR" >&2
		echo "Use one of: auto, gpu, cpu, tpu" >&2
		exit 1
		;;
esac

# Prefer latest compatible packages here to avoid pip backtracking after the
# accelerator-specific JAX/Flax install above.
python3 -m pip install --upgrade \
	pillow \
	clu \
	tensorflow \
	tensorflow-datasets \
	matplotlib \
	orbax-checkpoint \
	ml-dtypes \
	tensorstore \
	diffusers \
	dm-tree \
	cached_property \
	wandb