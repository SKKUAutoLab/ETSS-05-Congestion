pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install pip==24.0 cython ninja setuptools==75.0 six packaging
pip install numpy==1.26.4 scikit-learn scikit-image opencv-python h5py pyyaml gdown ftfy regex yapf==0.40.1 yacs easydict tqdm einops tensorboard tensorboardX imageio imageio-ffmpeg pandas matplotlib termcolor thop tabulate timm==1.0.19 efficientnet_pytorch transformers==4.40 rope nni ipdb
cd kernels/selective_scan 
pip install -e .
cd ../..
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
