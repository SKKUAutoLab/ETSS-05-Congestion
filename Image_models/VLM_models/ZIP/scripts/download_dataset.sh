mkdir -p data
cd data
wget https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/sha.zip
wget https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/shb.zip
wget https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/qnrf.zip
wget https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/qnrf.z01
wget https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.zip
wget https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z01
wget https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z02
wget https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z03
wget https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z04
wget https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z05
wget https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z06
wget https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z07
wget https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z08
unzip sha.zip
unzip shb.zip
7z x qnrf.zip
7z x nwpu.zip
cd ..
