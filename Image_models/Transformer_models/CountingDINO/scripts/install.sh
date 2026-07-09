pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install pip==24.0 cython ninja setuptools==65.0 six packaging
pip install -r requirements.txt
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
mkdir -p third_party
cd third_party
git clone https://github.com/facebookresearch/detectron2
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git
cd ..
git clone --recursive https://github.com/facebookresearch/CutLER
cd CutLER
pip install -r requirements.txt
cd ../..
