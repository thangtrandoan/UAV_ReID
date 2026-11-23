# Stage 1 (GASNet) Jetson AGX Orin Evaluation Suite

Bộ script đánh giá đầy đủ cho **Stage 1 (GASNet)** trên **NVIDIA Jetson AGX Orin**, tuân thủ đặc tả trong `huong_dan_danh_gia_stage1_jetson.md`.

## Cấu trúc mã nguồn

- `eval_config.yaml`: Cấu hình tập trung đường dẫn 12 checkpoint (`vru`, `veri`, `vrai`, `vehicleid` × `resnet50ibn_rga`, `convnext_rga`, `convnext_bcgg`), bộ dữ liệu, tham số benchmark PyTorch/TensorRT và trực quan hóa.
- `00_audit_manifest.py`: Kiểm kê môi trường Jetson Orin (`nvpmodel`, nhiệt độ GPU, CUDA/cuDNN/TensorRT/PyTorch) và SHA256 + khớp khóa cho toàn bộ checkpoint -> `00_env_and_manifest/`.
- `01_eval_accuracy.py`: Đánh giá độ chính xác (`mAP`, `Rank-1`, `Rank-5`, `Rank-10`, `mINP`) trên VeRi-776, VehicleID (`test_list_{800,1600,2400}.txt` × 10 seeds `[42..51]`), VRU (`Small/Medium/Large`), VRAI (`Val 1000 IDs`), và đối chiếu `PyTorch FP32` vs `PyTorch FP16` vs `TensorRT FP16` -> `01_accuracy_benchmarks/`.
- `02_verify_bcgg_folding.py`: Kiểm chứng tương đương toán học trước/sau `to_deploy()` của `BCGG` trên trọng số thật ở cấp Tensor (32 ảnh thật, `ga1..ga4` & `z(x)`) và cấp Truy hồi (`rank1_flips`) -> `02_bcgg_deploy_equivalence/`.
- `03a_static_complexity.py`, `03b_bench_pytorch.py`, `03c_export_and_bench_trt.py`: Đo `#Params`, `Gate #Params` (chỉ 4 cổng `ga1..ga4`), `GFLOPs`/`GMACs`, độ trễ PyTorch (`FP32` & `FP16`, `BS ∈ {1,2,4,8,32}`, breakdown `BS=1`), xuất ONNX opset 17 + `trtexec --fp16` (có `--timingCacheFile`, `builder_opt_level: 2`, `avg_timing: 2`, và tái sử dụng engine theo `SHA256`) + đo End-to-End 1 ảnh cắt qua thư viện 8.000 vector -> `03_efficiency_and_latency_orin/`.
- `04_bench_gate_scaling.py`: Đo riêng khối cổng 384 kênh ở `14×14`, `16×16`, `24×24`, `BS ∈ {1, 8}`, `FP32` & `FP16` -> `04_gate_scaling_ablation/`.
- `05_generate_visuals.py`: Xuất ảnh Top-5 retrieval (gồm cả ca failure), heatmap chú ý thật (`assert attn is not None`) và biểu đồ `delta_map`, `latency_fp32_fp16_trt`, `accuracy_vs_latency_pareto` ở `.png` (300 DPI) + `.pdf` -> `05_visualizations/`.
- `run_all.py`: Chạy toàn bộ pipeline tuần tự theo từng subprocess độc lập.

## Cách chạy trên Jetson AGX Orin

Khóa xung trước khi đo tốc độ:

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

Chạy toàn bộ pipeline (chấp nhận bỏ qua nếu chưa đủ 12 checkpoint hoặc chưa giải nén `VehicleID_V1.0.zip`):

```bash
python3 scripts/eval_stage1/run_all.py --config scripts/eval_stage1/eval_config.yaml
```

Hoặc chạy chế độ nghiêm ngặt (bắt buộc đủ 12 checkpoint và đủ 4 bộ dữ liệu, kèm khóa xung tự động):

```bash
python3 scripts/eval_stage1/run_all.py --config scripts/eval_stage1/eval_config.yaml --strict --lock-clocks
```

Chạy riêng từng bước (ví dụ):

```bash
python3 scripts/eval_stage1/00_audit_manifest.py
python3 scripts/eval_stage1/01_eval_accuracy.py
python3 scripts/eval_stage1/02_verify_bcgg_folding.py
python3 scripts/eval_stage1/03a_static_complexity.py
python3 scripts/eval_stage1/03b_bench_pytorch.py --lock-clocks
python3 scripts/eval_stage1/03c_export_and_bench_trt.py --lock-clocks
python3 scripts/eval_stage1/01_eval_accuracy.py --trt-only
python3 scripts/eval_stage1/04_bench_gate_scaling.py --lock-clocks
python3 scripts/eval_stage1/05_generate_visuals.py
```

## Treo máy chạy tự động & tự đẩy kết quả lên GitHub (xem từ máy khác)

Khi cần **treo máy Jetson** và lấy kết quả trên máy khác (không chung mạng LAN), dùng `run_unattended.sh`. Script sẽ tự chạy từng bước trong `tmux`, ghi log + `RUN_STATUS.json`, bỏ qua file nặng (`engines/`, `*.onnx`, `*.engine`, `*.cache`) nhờ `.gitignore`, và tự `git commit && git push` sau **từng bước**.

1. **Lưu credential GitHub 1 lần** (nếu máy chưa lưu token/SSH):

```bash
git config --global credential.helper store
git push --dry-run origin main
```

2. **Bật chế độ treo máy + khóa xung + tự đẩy kết quả (1 lệnh):**

```bash
./scripts/eval_stage1/run_unattended.sh --lock-clocks --tmux --include-code
```

- Muốn đẩy lên nhánh riêng (vd `results/stage1-orin`) để giữ sạch `main`: thêm `--branch results/stage1-orin`.
- Khi ở máy khác: mở repo trên GitHub xem trực tiếp `eval_results_stage1/00_env_and_manifest/RUN_STATUS.json` và `run_stage1.log`, hoặc chạy `git pull` để kéo toàn bộ bảng `.csv`, `.json` và hình `.png`/`.pdf` về.

## Chuẩn bị dữ liệu & checkpoint bổ sung (nếu thiếu)

- Nếu chưa giải nén `data/VehicleID_V1.0.zip` thành `data/VehicleID_V1.0/`, giải nén bằng mật khẩu dataset trước khi chạy với `--strict`.
- Đặt đủ 12 checkpoint đúng theo đường dẫn khai báo trong `scripts/eval_stage1/eval_config.yaml`.
