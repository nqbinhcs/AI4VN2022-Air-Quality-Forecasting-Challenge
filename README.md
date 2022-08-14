# AI4VN 2022 - Air Quality Forecasting Challenge

Cấu trúc repo
```
.
├── README.md
├── air_forecaster
│   ├── models
│   │   ├── ...chứa các class model
│   ├── trainer.py 
│   └── utils
│       ├── dataloader.py
│       └── getter.py
├── configs
│   └── ...chứa các file config .yaml 
├── data
│   ├── final-output
│   ├── final-results
│   ├── public-test
│   │   └── raw # chứa tập public test của BTC
│   └── train
│       └── raw # chứa tập train của BTC
├── requirements.txt
├── saved # lưu file weights ở mỗi lần train
├── scripts
│   └── ... các script (.sh) để chạy
├── setup.py
├── test.py
└── tools
    ├── inference.py
    ├── inference_drop_empty.py
    ├── preprocess_data.py
    └── train.py
```

Download dữ liệu cần thiết (data train và public test) rồi để vào các folder ở trên.

Tiến hành cài đặt các packages
```
pip3 install -r requirements.txt
python3 -m pip install -e .
```

Sau đó, có thể chỉnh sửa các script hoặc config theo ý muốn rồi tiến hành train/test tùy ý.


