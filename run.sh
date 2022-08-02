python src/train.py 
python src/inference.py
cd results
rm prediction.zip
zip -r prediction.zip *
cd ..
