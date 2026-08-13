mkdir -p weights
cd weights
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
cd ..
gdown --folder "https://drive.google.com/drive/folders/12Ak9uIyLHkRo59zCbEZqXrRe0kP5iz_S" -O ./weights --remaining-ok
