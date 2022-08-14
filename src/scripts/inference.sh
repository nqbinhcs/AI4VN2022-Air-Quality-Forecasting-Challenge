python3 -W ignore tools/inference.py --weights_folder get_latest_folder --results_folder data/final-results

cd data/final-results
rm prediction.zip
zip -r prediction.zip *
cd ..
cp final-results/prediction.zip final-output/