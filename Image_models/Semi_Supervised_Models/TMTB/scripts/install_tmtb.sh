pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install pip==24.0 cython ninja setuptools==75.0 six packaging
pip install -r requirements.txt
cd kernels/selective_scan 
pip install -e .
cd ../..
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
