pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install pip==24.0 cython ninja setuptools==75.0 six packaging
cd src/open-r1-multimodal
pip install -e ".[dev]"
pip install -e .
pip install wandb==0.18.3 tensorboardx qwen_vl_utils
cd ../..
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install math-verify peft==0.17.1
