# Inference using xgboost (current best on private test)
python3 -W ignore tools/inference_private.py \
--phase private-test \
--weights_folder saved/xgboost_best_0 \
--results_folder data/final-results \
--inference_method idw

# Inference using LSTM
# python3 -W ignore tools/inference_private.py \
# --phase private-test \
# --weights_folder saved/lstm_4 \
# --results_folder data/final-results \
# --inference_method idw 

cd data/final-results
rm prediction.zip
zip -rq prediction.zip *
cd ..
mkdir final-output
cp final-results/prediction.zip final-output/