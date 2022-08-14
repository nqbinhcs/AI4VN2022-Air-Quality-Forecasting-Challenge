# run install current package
python3 -m pip install -e .

# preprocess data
python3 -W ignore tools/preprocess_data.py \
--data_train_folder_path="data/train/raw/" \
--preprocessed_data_train_folder_path="data/train/preprocessed/" \
--public_test_folder_path="data/public-test/raw/" \
--preprocessed_public_test_folder_path="data/public-test/preprocessed/"

# train
python3 tools/train.py --config_path configs/best_public_test.yaml --saved_runs_folder saved

# inference
python3 -W ignore tools/inference.py --weights_folder get_latest_folder --results_folder data/final-results

# zip final results
cd data/final-results
rm prediction.zip
zip -r prediction.zip *
cd ..
cp final-results/prediction.zip final-output/