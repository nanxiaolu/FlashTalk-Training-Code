# Environment setup

The recipe below is what we use in production. Anything else is
unsupported.

| Component | Version |
|---|---|
| OS | Linux x86_64 |
| CUDA driver | 12.1+ |
| Python | 3.10.18 |
| PyTorch | 2.4.1 + cu121 |
| Hardware | 8× A100/A800 80GB or 8× H100/H800 80GB |

## Step 1 — create the conda env

```bash
conda create -n flashtalk_cxy python=3.10.18 -y
conda activate flashtalk_cxy
```

## Step 2 — install PyTorch (cu121)

```bash
pip install \
  torch==2.4.1+cu121 \
  torchvision==0.19.1+cu121 \
  torchaudio==2.4.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

## Step 3 — pip mirror & cache (optional but recommended for CN users)

You can write your mirror + cache once and pip will pick them up forever.
Make sure you only keep **one** pip config (we ran into priority conflicts
between `/root/.pip/pip.conf` and `/root/.config/pip/pip.conf`).

```bash
mkdir -p /root/.pip
cat > /root/.pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
cache-dir = /root/group-shared/digital-human/chenxiaoyong/.cache/pip

[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

(Replace the `cache-dir` with whatever path makes sense on your machine.
Pointing it at a large shared FS speeds up repeated installs.)

## Step 4 — project dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` is pinned and intentionally short. The notable pins:

* `setuptools<81` — `librosa==0.9.2` still imports the legacy
  `pkg_resources` API, which `setuptools>=81` removed.
* `numpy==1.26.4` — `insightface==0.7.3` ships wheels that try to drag
  numpy 2.x in transitively, but every downstream consumer here requires
  numpy 1.x. We pin both numpy and `opencv-python-headless==4.10.0.82`
  (the last numpy<2-compatible release).

## Step 5 — install flash-attn (mandatory, but installed separately)

`flash-attn` is **required** at runtime but is omitted from
`requirements.txt` because:

1. It needs `--no-build-isolation` to find the local PyTorch headers.
2. The right version depends on your GPU generation.

```bash
# A-series (A100/A800/A40/A30/A6000 ...): use flash-attn v2.
pip install flash-attn==2.7.4.post1 --no-build-isolation

# H-series (H100/H800/H200 ...): flash-attn v3 is significantly faster and
# is what we recommend, but the install procedure is more involved — follow
# the upstream README at https://github.com/Dao-AILab/flash-attention .
```

If the build fails or takes >30 min, the most common causes are:

* `nvcc` mismatched with `torch.version.cuda`. Check with `nvcc --version`
  and `python -c "import torch; print(torch.version.cuda)"`. They must
  agree on the major version (12.x).
* Not enough RAM for `ninja` parallel build. Use
  `MAX_JOBS=4 pip install flash-attn==... --no-build-isolation`.

## Step 6 — quick sanity check

```bash
python - <<'EOF'
import torch, flash_attn, transformers, diffusers, lmdb, insightface
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("flash_attn:", flash_attn.__version__)
print("transformers:", transformers.__version__)
EOF
```

If all imports succeed and CUDA is `True`, the env is ready.
