python3 -W ignore tools/preprocess_data.py \
--type="private-test" \
--method='interpolate' \
--data_train_folder_path="data/private-train/raw/" \
--preprocessed_data_train_folder_path="data/private-train/preprocessed/interpolate-method" \
--public_test_folder_path="data/public-test/raw/" \
--preprocessed_public_test_folder_path="data/public-test/preprocessed/"