pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install pip==24.0 cython ninja setuptools==75.0 six packaging
pip install -r requirements.txt
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
conda install mpi4py -y
pip install git+https://github.com/openai/whisper.git
