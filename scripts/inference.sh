# python3 -W ignore tools/inference.py --weights_folder get_latest_folder --results_folder data/final-results --inference_method idw

# XGBOOST
# python3 -W ignore tools/inference_private.py \
# --phase private-test \
# --weights_folder saved/xgboost_best_0 \
# --results_folder data/final-results \
# --inference_method idw

# LSTM
# To inference LSTM, change function "load_data(data_path, test_size, is_full=False)" to "load_data(data_path, test_size, is_full=True)" in data loader
python3 -W ignore tools/inference_private.py \
--phase private-test \
--weights_folder saved/lstm_4 \
--results_folder data/final-results \
--inference_method idw 

# CATBOOST
# python3 -W ignore tools/inference_private.py --weights_folder saved/catboost_2 --results_folder data/final-results --inference_method idw


cd data/final-results
rm prediction.zip
zip -rq prediction.zip *
cd ..
mkdir final-output
cp final-results/prediction.zip final-output/