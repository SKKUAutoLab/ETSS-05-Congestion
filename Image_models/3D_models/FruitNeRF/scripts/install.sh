export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda-11.8/
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio==0.3.2
pip install viser==0.1.24
python -m pip install -e .
mkdir -p segmentation
cd segmentation
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git grounded_sam
cd grounded_sam
git checkout fe24
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install --upgrade diffusers[torch]
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
cd ..
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
pip install segment-anything-hq
cd ..
pip install tyro==0.8.12 trimesh==4.4.7
