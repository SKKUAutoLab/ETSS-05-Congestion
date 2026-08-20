pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pip==24.0 cython ninja setuptools==59.5.0 six packaging
pip install -r requirements.txt
cd CLIP
python setup.py develop
cd ..
