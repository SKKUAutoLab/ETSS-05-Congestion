mkdir -p datasets
cd datasets
gdown https://drive.google.com/u/0/uc?id=1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S
unzip FSC147_384_V2.zip -d FSC147
rm -rf FSC147_384_V2.zip
cd ..
mkdir -p annotations
cd annotations
wget https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/refs/heads/master/data/Train_Test_Val_FSC_147.json
wget https://github.com/cvlab-stonybrook/LearningToCountEverything/raw/refs/heads/master/data/annotation_FSC147_384.json
cd ..
