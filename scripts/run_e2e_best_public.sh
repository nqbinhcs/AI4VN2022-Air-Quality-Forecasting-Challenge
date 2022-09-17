# preprocess data
python3 -W ignore tools/preprocess_data.py \
--phase public-test \
--method mean \
--train_folder_path="data/public-train/raw" \
--preprocessed_train_folder_path="data/public-train/preprocessed/mean-method" \
--test_folder_path="data/public-test/raw" \
--preprocessed_test_folder_path="data/public-test/preprocessed/mean-method"

# train
python3 tools/train.py \
--config_path configs/best_public_test.yaml \
--saved_runs_folder saved

# inference
python3 -W ignore tools/inference.py \
--weights_folder get_latest_folder \
--results_folder data/final-results \
--inference_method idw

# zip final results
cd data/final-results
rm prediction.zip
zip -r prediction.zip *
cd ..
cp final-results/prediction.zip final-output/