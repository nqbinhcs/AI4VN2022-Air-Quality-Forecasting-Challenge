# AI4VN 2022 - Air Quality Forecasting Challenge
## Team: LQDBD
- Nguyễn Tiến Hưng
- Nguyễn Quang Bình
- Nguyễn Hữu Đạt
- Lê Minh Tú

## Hướng dẫn reproduce lại kết quả
Đầu tiên, tiến hành build Docker image
```
docker build -t <tên image> .
```

Sau khi quá trình build Docker image hoàn tất, vui lòng chỉnh sửa file `run_container.sh` để tiến hành mount các folder chứa dữ liệu cần thiết:
- folder data-train
- folder public-test
- folder chứa file zip kết quả prediction (`prediction.zip`)

Lưu ý: khi mount cần mount trực tiếp vào folder cha, ví dụ khi data trên máy được lưu có cấu trúc:
```
ai4vn-aqf
├── data-train
    ├── input/
    ├── output/
    └── location.csv
```
Thì cần mount như sau: `/ai4vn-aqf/data-train/:/data/train/raw/`

Khi đã hoàn tất mount folder thì có thể tiến hành chạy docker container
```
sh run_container.sh
```



