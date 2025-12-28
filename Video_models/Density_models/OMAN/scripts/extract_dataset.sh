cd data
zip -FF videos.zip --out combine.zip
unzip combine.zip -d Sense
unzip annotations.zip -d Sense
mv Sense/annotations Sense/label_list_all
rm -rf combine.zip
cd ..