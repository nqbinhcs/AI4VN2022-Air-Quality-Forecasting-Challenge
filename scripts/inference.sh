# python3 -W ignore tools/inference.py --weights_folder get_latest_folder --results_folder data/final-results --inference_method idw
# python3 -W ignore tools/inference_full.py --weights_folder saved/lstm_9 --results_folder data/final-results
python3 -W ignore tools/inference_private.py --weights_folder saved/xgboost_best_0 --results_folder data/final-results --inference_method idw
# python3 -W ignore tools/inference_full.py --weights_folder saved/lstm_4 --results_folder data/final-results 

cd data/final-results
rm prediction.zip
zip -rq prediction.zip *
cd ..
mkdir final-output
cp final-results/prediction.zip final-output/