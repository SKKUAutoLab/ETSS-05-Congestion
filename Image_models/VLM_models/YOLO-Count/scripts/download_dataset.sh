mkdir -p data/FSC
cd data/FSC
gdown https://drive.google.com/u/0/uc?id=1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S
unzip FSC147_384_V2.zip
cd ../..
python -m scripts.download_oimgv7
python -m scripts.download_o365
