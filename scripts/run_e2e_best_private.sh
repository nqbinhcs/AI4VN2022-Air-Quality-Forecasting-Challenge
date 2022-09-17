# preprocess data
python3 -W ignore tools/preprocess_data.py \
--phase private-test \
--method mean \
--train_folder_path="data/private-train/raw" \
--preprocessed_train_folder_path="data/private-train/preprocessed/mean-method" \
--test_folder_path="data/private-test/raw" \
--preprocessed_test_folder_path="data/private-test/preprocessed/mean-method"

# train
python3 tools/train.py \
--config_path configs/best_private_test.yaml \
--saved_runs_folder saved

# inference
python3 -W ignore tools/inference_private.py \
--weights_folder get_latest_folder \
--results_folder data/final-results \
--inference_method idw

# zip final results
cd data/final-results
rm prediction.zip
zip -rq prediction.zip *
cd ..
mkdir final-output
cp final-results/prediction.zip final-output/