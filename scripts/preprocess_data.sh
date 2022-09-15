python3 -W ignore tools/preprocess_data.py \
--phase private-test \
--method mean \
--train_folder_path="data/private-train/raw" \
--preprocessed_train_folder_path="data/private-train/preprocessed/mean-method" \
--test_folder_path="data/private-test/raw" \
--preprocessed_test_folder_path="data/private-test/preprocessed/mean-method"