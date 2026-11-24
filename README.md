# tinyclassic-esp32-bench

Đo hiệu năng của các mô hình học máy cổ điển sau khi được chuyển sang mã C/C++ bằng emlearn, micromlgen và m2cgen, chạy trên ESP32, ESP32-C3 và ESP32-S3.

## Câu hỏi nghiên cứu

- RQ1. Mỗi công cụ chuyển được những mô hình nào?
- RQ2. Mô hình sau khi chuyển có cho cùng nhãn với Python không? Chiếm bao nhiêu flash, RAM và mất bao lâu để suy luận?
- RQ3. FPU và kiểu số (`float`, `double`) ảnh hưởng thế nào đến thời gian suy luận?

## Dữ liệu

| Bộ dữ liệu | Train | Test | Đặc trưng | Lớp | Chia train/test | Nguồn |
| --- | --- | --- | --- | --- | --- | --- |
| `occupancy` | 8.143 | 12.417 | 5 | 2 | Train: `datatraining.txt`; test: `datatest.txt` + `datatest2.txt` | UCI id 357 |
| `smoke` | 43.842 | 18.788 | 12 | 2 | Mỗi phiên ghi: 70% đầu train, 30% cuối test | `Blatts01/ai-smokedetector`, `data/final.csv` (MIT) |
| `gasdrift` | 9.546 | 4.364 | 128 | 6 | Batch 1–7 train, batch 8–10 test | UCI id 224 (CC BY 4.0, chỉ dùng cho nghiên cứu) |
| `har` | 7.352 | 2.947 | 561 | 6 | Chia sẵn theo người tham gia | UCI id 240 |

- Occupancy: dùng Temperature, Humidity, Light, CO2, HumidityRatio.
- Smoke: bỏ `CNT` và `UTC`, nhãn là `Fire Alarm`. Phiên ghi mới bắt đầu khi khoảng cách giữa hai giá trị `UTC` liên tiếp lớn hơn 60 s (dữ liệu có 5 phiên).
- Chuẩn hóa z-score với tham số tính trên tập train; nhãn được mã hóa về 0..K−1.
- Checksum của các file tải về nằm trong `data/checksums.json`.

## Mô hình

| Tên | Mô hình | Cấu hình |
| --- | --- | --- |
| `dt5`, `dt10` | Decision Tree | `max_depth` 5, 10 |
| `rf10`, `rf50` | Random Forest | 10, 50 cây, `max_depth=8` |
| `gnb` | Gaussian NB | mặc định |
| `svc` | SVC, kernel RBF | `gamma = 1 / (n_features * X_train.var())` |
| `xgb` | XGBoost | 50 vòng, `max_depth=4`, `base_score=0.5`, `num_parallel_tree=1` |

`random_state=42` cho mọi mô hình. Nhãn tham chiếu là kết quả `predict` của Python trên tập test dạng `float32`.

## Công cụ chuyển đổi

| Tên | Công cụ | Mô hình |
| --- | --- | --- |
| `emlearn` | emlearn 0.23.2 | dt, rf, gnb |
| `micromlgen` | micromlgen 1.1.28 | dt, rf, gnb, svc, xgb |
| `m2cgen` | m2cgen 0.10.0 | dt, rf, svc, xgb |
| `m2cgen_float` | m2cgen 0.10.0, chuyển sang `float` | dt, rf, svc, xgb |

Mỗi cấu hình (bộ dữ liệu, mô hình, công cụ) được sinh vào `generated/<dataset>/<model>/<tool>/`:

- `model_info.h`: số đặc trưng, số lớp, mã cấu hình.
- `model.h` hoặc `model.c`: mã do công cụ sinh ra.
- `adapter.c` hoặc `adapter.cpp`: hàm `int bench_predict(const float *x)` dùng chung cho máy tính và board.
- `meta.json`: trạng thái chuyển đổi.

Ngoài ra mỗi bộ dữ liệu có cấu hình `baseline/baseline`, trong đó `bench_predict` trả về 0. Kích thước của mô hình được tính bằng cách trừ đi kích thước firmware của cấu hình này.

Quy ước khi chuyển:

- emlearn: cây dùng `method='inline'`, `dtype='float'`; Random Forest giữ `leaf_bits` mặc định.
- micromlgen, Gaussian NB: gán `sigma_ = var_` trước khi gọi `port()`.
- micromlgen, XGBoost: không chuyển được bài toán 2 lớp; ghi nhận là lỗi chuyển đổi.
- m2cgen: `sys.setrecursionlimit(100000)`.
- m2cgen: đầu vào được chép sang mảng `double` trong `bench_predict`. SVC 2 lớp lấy dấu của `score()`; SVC nhiều lớp bỏ phiếu one-vs-one trên các cặp (0,1), (0,2), …; các mô hình khác lấy argmax.
- m2cgen_float: thay `double` bằng `float`, `exp` bằng `expf`, `pow` bằng `powf`, thêm hậu tố `f` cho hằng số. `host_check` đếm số cảnh báo `-Wdouble-promotion` của `model.c`.

## Phần cứng

| Profile | Target | CPU | Xung nhịp |
| --- | --- | --- | --- |
| `esp32_240`, `esp32_160` | `esp32` | Xtensa LX6, FPU single | 240, 160 MHz |
| `esp32c3_160` | `esp32c3` | RISC-V, không FPU | 160 MHz |
| `esp32s3_240`, `esp32s3_160` | `esp32s3` | Xtensa LX7, FPU single | 240, 160 MHz |

Board: ESP32-DevKitC (WROOM-32), ESP32-C3 DevKit, ESP32-S3-DevKitC, mỗi board có ít nhất 4 MB flash.

## Firmware

Project ESP-IDF v6.1.1 trong `firmware/`, dùng chung cho cả ba target.

| File | Nội dung |
| --- | --- |
| `sdkconfig.defaults` | `-O2` (`COMPILER_OPTIMIZATION_PERF`), 240 MHz, console UART 921600 baud, flash 4 MB, bảng phân vùng riêng |
| `sdkconfig.defaults.esp32c3` | 160 MHz, tắt console phụ qua USB Serial/JTAG |
| `sdkconfig.defaults.esp32s3` | tắt console phụ qua USB Serial/JTAG |
| `sdkconfig.160mhz` | 160 MHz, dùng cho `esp32_160` và `esp32s3_160` |
| `partitions.csv` | một phân vùng ứng dụng 0x3F0000 byte |
| `main/Kconfig.projbuild` | kênh kết nối (UART0 hoặc USB Serial/JTAG), số lần chạy khởi động, số lần đo |

Mỗi profile có thư mục build riêng `build/<profile>/`. Cấu hình cần đo được chép vào `build/<profile>/model/` rồi build lại.

Giao thức với máy tính (little-endian, mỗi lần một yêu cầu):

| Máy tính gửi | Board trả lời |
| --- | --- |
| `H` | dòng `BENCH READY id=… target=… mhz=… nf=… nc=… warmup=… repeats=…` |
| `S` + N_FEATURES × float32 | `R`, int32 nhãn, uint32 trung vị chu kỳ, uint32 chu kỳ nhỏ nhất, uint32 số lần đo |
| `E` | `M`, uint32 kích thước stack, uint32 stack trống tối thiểu (byte), uint32 heap trống, uint32 heap trống tối thiểu |

Với mỗi mẫu, board gọi `bench_predict` 3 lần không đo, rồi 7 lần có đo bằng `esp_cpu_get_cycle_count()`. Nếu lần gọi đầu tiên vượt 20.000.000 chu kỳ thì chỉ đo lần đó. Task đo chạy trên nhân cuối cùng của chip.

## Chỉ số

| Chỉ số | Nguồn |
| --- | --- |
| Trạng thái: chuyển được, biên dịch được, vừa bộ nhớ | `meta.json`, `results/host_check.csv`, `results/size.csv` |
| Tỉ lệ nhãn trùng với Python | `results/host_check.csv` (máy tính), `results/raw/` (board) |
| Accuracy, macro-F1 trên board | `results/raw/` |
| Flash của mô hình | kích thước `bench.bin` trừ cấu hình `baseline` |
| RAM tĩnh của mô hình | `.dram0.data` + `.dram0.bss` trong ELF, trừ cấu hình `baseline` |
| Stack | kích thước stack trừ stack trống tối thiểu |
| Thời gian suy luận | trung vị và phân vị 95 của các trung vị theo mẫu, đơn vị chu kỳ và µs |

## Cài đặt

```bash
uv venv -p 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt
```

Trên macOS, xgboost cần `libomp` (`brew install libomp`).

ESP-IDF v6.1.1 cài theo hướng dẫn của Espressif. Các bước `build` và `run` chạy trong shell đã chạy `. $IDF_PATH/export.sh`.

## Quy trình

```bash
python -m bench.download        # data/raw, data/checksums.json
python -m bench.split           # data/processed
python -m bench.train           # work/models, results/python_models.csv
python -m bench.convert         # generated/
python -m bench.host_check      # results/host_check.csv
python -m bench.env             # results/env.json
python -m bench.build --profile esp32_240 esp32_160 esp32c3_160 esp32s3_240 esp32s3_160
python -m bench.run --profile esp32_240 --port <cổng serial>
python -m bench.analyze         # results/summary.csv
```

Các bước nhận `--dataset`, `--model`, `--tool` để chạy một phần. `bench.run` bỏ qua cấu hình đã có kết quả, trừ khi có `--force`; `--limit N` chỉ gửi N mẫu test đầu tiên.

## Cấu trúc thư mục

```text
bench/          pipeline Python
host/           chương trình chạy mô hình trên máy tính
firmware/       project ESP-IDF
pilot/          thử nghiệm ban đầu với dữ liệu tổng hợp
data/           checksums.json; raw/ và processed/ không commit
generated/      mã sinh ra (không commit)
work/           mô hình đã huấn luyện (không commit)
build/          thư mục build theo profile (không commit)
results/        kết quả
```

## Tiến độ

- [x] Tải và chia dữ liệu
- [x] Huấn luyện
- [x] Chuyển đổi và kiểm tra trên máy tính
- [ ] Build firmware cho 5 profile
- [ ] Đo trên 3 board
- [ ] Tổng hợp kết quả
