# ĐẶC TẢ HỆ THỐNG ĐÁNH GIÁ & BENCHMARK STAGE 1 (GASNET) TRÊN JETSON AGX ORIN

Tài liệu này là bản hướng dẫn kỹ thuật chi tiết để AI coding agent trên máy **NVIDIA Jetson AGX Orin** viết script và chạy toàn bộ thực nghiệm cho **Giai đoạn 1 (GASNet)** của khóa luận *"Nghiên cứu và phát triển thuật toán tái định danh phương tiện theo thời gian thực ứng dụng trên UAV"*.

## 0. Quy ước đầu vào & Nguyên tắc bắt buộc

### Danh sách 12 checkpoint đầu vào
Hệ thống có 12 checkpoint đã huấn luyện/tinh chỉnh trên 4 miền dữ liệu (`vru`, `veri`, `vrai`, `vehicleid`) × 3 cấu hình (`resnet50ibn_rga`, `convnext_rga`, `convnext_bcgg`). Toàn bộ script cần đọc đường dẫn checkpoint từ một tệp cấu hình tập trung `eval_config.yaml` (không hardcode đường dẫn).

### 5 Nguyên tắc kỹ thuật bắt buộc (Chống lỗi phản biện hội đồng)
1. **Khóa xung phần cứng trước khi đo tốc độ:** Trước mọi phiên đo độ trễ, bắt buộc chạy `sudo nvpmodel -m 0 && sudo jetson_clocks` và ghi lại chế độ công suất + nhiệt độ GPU vào log.
2. **Đồng nhất mô hình giữa bảng độ chính xác và bảng tốc độ:** Mô hình baseline phải là **`ResNet50-IBN + RGA`** ở cả bảng độ chính xác lẫn bảng tham số/FLOPs/độ trễ (không dùng `ResNet-50` thường để đo tốc độ rồi ghép với điểm của `ResNet50-IBN`).
3. **Reset bộ nhớ GPU sạch trước mỗi cấu hình:** Trước khi đo `Peak VRAM` của từng mô hình, bắt buộc giải phóng mô hình cũ (`del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()`).
4. **Cấm fallback giả khi vẽ Heatmap:** Khi trích xuất bản đồ chú ý (`last_attention`), bắt buộc `assert attn is not None`. Nếu hook không lấy được tensor thì dừng chương trình báo lỗi, tuyệt đối không vẽ bản đồ Gaussian giả.
5. **Mọi kết quả phải xuất ra đồng thời 2 định dạng:** `.csv` (số liệu thô/tổng hợp để vẽ biểu đồ và đưa vào bảng LaTeX) và `.json` (metadata, hạt giống ngẫu nhiên, số lượng truy vấn hợp lệ sau lọc).

## 1. Cấu trúc thư mục đầu ra (`eval_results_stage1/`)

AI agent cần tổ chức mã nguồn đánh giá (ví dụ trong `scripts/eval_stage1/`) và xuất toàn bộ kết quả theo đúng 6 thư mục dưới đây:

```text
eval_results_stage1/
├── 00_env_and_manifest/
├── 01_accuracy_benchmarks/
├── 02_bcgg_deploy_equivalence/
├── 03_efficiency_and_latency_orin/
├── 04_gate_scaling_ablation/
└── 05_visualizations/
```

## 2. Yêu cầu chi tiết cho từng thư mục

### Thư mục 1: `00_env_and_manifest/` — Môi trường Orin & Kiểm kê 12 Checkpoint
- **Script đề xuất:** `00_audit_manifest.py`
- **Nhiệm vụ:**
  1. Ghi thông tin hệ thống: mẫu Jetson Orin, JetPack/L4T, CUDA, cuDNN, TensorRT, PyTorch, Python, trạng thái `nvpmodel`, nhiệt độ GPU.
  2. Quét toàn bộ 12 checkpoint: tính mã `SHA256`, tự động nhận diện `backbone` (`resnet50_ibn` vs `dinov3_convnext`), `gate` (`rga` vs `bcgg`), siêu tham số BCGG (`hidden`, `groups`, `kernel_size`, `channel_kernel`, `inject_value`), số lớp `num_classes`, số key khớp `matched/total` và liệt kê rõ mọi key thiếu/dư (nếu có).
- **Tệp đầu ra bắt buộc:**
  - `env_info.json`
  - `checkpoints_manifest.json`
  - `checkpoints_manifest.csv`

### Thư mục 2: `01_accuracy_benchmarks/` — Đánh giá Độ chính xác trên 4 Bộ dữ liệu
- **Script đề xuất:** `01_eval_accuracy.py`
- **Độ đo bắt buộc cho mọi bộ dữ liệu:** `mAP`, `Rank-1`, `Rank-5`, `Rank-10`, `mINP` (tính theo phần trăm `%`, làm tròn 2 chữ số thập phân ở bảng tổng hợp, giữ full precision trong JSON).
- **Giao thức chuẩn cho từng bộ dữ liệu:**
  1. **VeRi-776 (`Full Test`):** 1.678 query, 11.579 gallery (200 ID). Lọc bỏ mọi ảnh gallery có cùng `vehicle_id` và cùng `camera_id` với ảnh query. Ghi rõ số query hợp lệ sau lọc.
  2. **VehicleID (`VehicleID_V1.0`):** Đọc trực tiếp 3 danh sách chuẩn `test_list_800.txt`, `test_list_1600.txt`, `test_list_2400.txt` (không dùng định dạng cặp Brno). Với mỗi split, chạy **10 lượt rút gallery ngẫu nhiên** (mỗi ID chọn 1 ảnh làm gallery, các ảnh còn lại làm query), dùng chung đúng bộ 10 seed `[42, 43, ..., 51]` cho cả 3 mô hình. Báo cáo `mean ± std` trên 10 lượt.
  3. **VRU:** 3 split `Small (1200)`, `Medium (2400)`, `Large (8000)`. Mỗi ID rút 1 ảnh làm gallery với cùng `seed = 42` cho cả 3 mô hình.
  4. **VRAI (Tập kiểm định nội bộ 1.000 ID):** Đánh giá trên đúng 1.000 ID tách riêng khỏi 5.302 ID huấn luyện (`seed = 42`), áp luật lọc khác camera. **Bắt buộc ghi rõ cả `total_queries_before_filter` và `valid_queries_after_filter`** vào metadata.
- **Đối chiếu độ chính xác theo kiểu số (Precision Consistency):**
  - Chạy đánh giá cho cả 12 checkpoint ở `FP32` (hoặc `BF16` chuẩn) và chạy lại ít nhất trên **VeRi-776** và **VRAI** ở **`FP16`** (và **`TensorRT FP16`** sau khi build ở Thư mục 4) để chứng minh chuyển sang FP16/TensorRT không làm sụt mAP/Rank-1.
- **Tệp đầu ra bắt buộc:**
  - `accuracy_summary.csv` (mỗi hàng là 1 cặp `dataset_split × model`, các cột: `dataset, split, model, precision, valid_queries, gallery_size, mAP, R1, R5, R10, mINP`)
  - `vehicleid_10runs_raw.csv` (lưu chi tiết từng lượt chạy 1..10 cho 3 split × 3 mô hình)
  - `precision_comparison_fp32_fp16_trt.csv` (so sánh mAP/R1 giữa PyTorch FP32, PyTorch FP16 và TensorRT FP16)
  - `protocol_metadata.json` (ghi rõ kích thước ảnh tiền xử lý `Resize(224)` hay `Resize(256)+CenterCrop(224)`, hạt giống, số mẫu trước/sau lọc)

### Thư mục 3: `02_bcgg_deploy_equivalence/` — Kiểm chứng Tương đương Toán học `BCGG` trước & sau `to_deploy()`
- **Script đề xuất:** `02_verify_bcgg_folding.py`
- **Nhiệm vụ:** Khắc phục triệt để lỗi kiểm tra trên trọng số rỗng (`checkpoint=None`). **Bắt buộc nạp checkpoint `convnext_bcgg` đã huấn luyện** (chạy trên ít nhất 2 checkpoint: `vru` và `veri`).
  1. *Kiểm tra cấp Tensor:* Cho 1 batch 32 ảnh thật chạy qua `model.eval()` (chưa gộp) và `model.to_deploy().eval()` (đã gộp) ở cả `FP32` và `FP16`. Đo `max_abs_diff` và `mean_abs_diff` tại đầu ra từng cổng `ga1, ga2, ga3, ga4` và tại vector đặc trưng cuối `z(x) [960]`, kèm `min_cosine_sim` giữa hai bộ vector.
  2. *Kiểm tra cấp Truy hồi:* Chạy đánh giá `BCGG chưa gộp` vs `BCGG đã gộp` trên `VeRi-776` và `VRU Small`, ghi lại chênh lệch `mAP`, `Rank-1`, `mINP` và đếm chính xác số lượng truy vấn bị đổi hạng đầu (`rank1_flips`).
- **Tệp đầu ra bắt buộc:**
  - `bcgg_tensor_equivalence.csv`
  - `bcgg_retrieval_equivalence.csv`
  - `bcgg_equivalence_report.json` (kèm `SHA256` của checkpoint đã dùng)

### Thư mục 4: `03_efficiency_and_latency_orin/` — Đo Chi phí Tính toán & Tốc độ Thời gian thực trên Jetson AGX Orin
- **Script đề xuất:** `03a_static_complexity.py`, `03b_bench_pytorch.py`, `03c_export_and_bench_trt.py`
- **4 cấu hình bắt buộc đo song song trong cùng một phiên:**
  1. `resnet50ibn_rga`
  2. `convnext_rga`
  3. `convnext_bcgg_unfolded` (chưa gọi `to_deploy()`)
  4. `convnext_bcgg_folded` (đã gọi `to_deploy()`)
- **Phần A — Độ phức tạp tĩnh (`static_complexity.csv`):**
  - Với ảnh đầu vào `1×3×224×224`, đo: `total_params_M`, `gate_params_K` (chỉ đếm đúng 4 module `ga1, ga2, ga3, ga4`, không đếm nhầm `ChannelGate` trong `OSBlockFS`), `GFLOPs`, `GMACs`, và dung lượng file trọng số (MB).
- **Phần B — Độ trễ PyTorch trên Orin (`latency_pytorch_raw.csv`, `latency_pytorch_summary.csv`):**
  - Đo trên cả **2 kiểu số: `FP32` và `FP16`**, tại các cỡ batch **`BS ∈ {1, 2, 4, 8, 32}`** (`BS=1` cho 1 xe/khung, `BS=2,4,8` cho nhiều xe trong cùng khung hình UAV, `BS=32` cho nạp thư viện).
  - Giao thức đo: warm-up 50 lượt, đo 200 lượt (với `BS ≤ 8`) và 100 lượt (với `BS = 32`) dùng `torch.cuda.Event(enable_timing=True)` + `torch.cuda.synchronize()`.
  - Các chỉ số xuất ra: `p50_ms`, `p95_ms`, `mean_ms`, `std_ms`, `throughput_ips` (`1000 * BS / p50_ms`), `peak_vram_MiB`, tỷ lệ lượt đạt ngưỡng 30 FPS (`pct_under_33_33ms`) và 25 FPS (`pct_under_40ms`).
  - Phân rã thời gian ở `BS = 1`: đo riêng `backbone_ms` vs `gates_fs_head_ms`.
- **Phần C — Xuất ONNX & Đo tốc độ TensorRT FP16 + End-to-End (`latency_tensorrt_summary.csv`, `latency_e2e_summary.csv`):**
  - Xuất ONNX (`opset=17`, dynamic batch `1..32`) cho 3 cấu hình (`resnet50ibn_rga`, `convnext_rga`, `convnext_bcgg_folded`).
  - Build TensorRT engine **FP16** bằng `/usr/src/tensorrt/bin/trtexec --fp16` với profile `min=1, opt=1, max=32` (và thêm profile `opt=4` hoặc `opt=8` nếu cần).
  - Đo độ trễ thuần engine tại `BS ∈ {1, 2, 4, 8, 32}` (200 lượt sau warm-up): `p50_ms`, `p95_ms`, `throughput_ips`, `engine_size_MB`.
  - **Đo End-to-End 1 ảnh cắt (Chứng minh "thời gian thực"):** Đo trọn vòng đời từ ảnh cắt trên CPU (OpenCV BGR) $\rightarrow$ Resize + Normalize $\rightarrow$ H2D $\rightarrow$ TensorRT FP16 inference $\rightarrow$ D2H $\rightarrow$ L2 normalize + nhân ma trận cosine với thư viện 8.000 vectơ (VRU Large) + `argsort` lấy Top-10. Ghi tách biệt thời gian từng khâu và tổng `e2e_p50_ms`, `e2e_p95_ms`, `pct_under_33_33ms`.

### Thư mục 5: `04_gate_scaling_ablation/` — Đo riêng Khối Cổng (`RGA` vs `BCGG`) theo Độ phân giải
- **Script đề xuất:** `04_bench_gate_scaling.py`
- **Nhiệm vụ:** Đo độc lập khối cổng 384 kênh (tầng 3) tại 3 kích thước bản đồ đặc trưng `14×14` (ảnh 224), `16×16` (ảnh 256), `24×24` (ảnh 384), ở `BS = 1` và `BS = 8`, trên cả `FP32` và `FP16`.
- **4 biến thể cổng:** `RGA cố định (14×14)` (ghi `N/A` khi không tương thích ở `16×16`, `24×24`), `RGA khởi tạo lại theo cỡ`, `BCGG chưa gộp`, `BCGG đã gộp`.
- **Tệp đầu ra bắt buộc:** `gate_scaling_summary.csv` (cột: `spatial_size, precision, batch_size, gate_type, params_K, GFLOPs, p50_ms, p95_ms, peak_vram_MiB`).

### Thư mục 6: `05_visualizations/` — Hình ảnh Trực quan hóa & Biểu đồ cho Chương 5 (300 DPI + PDF)
- **Script đề xuất:** `05_generate_visuals.py`
- **Các nhóm hình cần xuất (lưu cả `.png` 300 DPI và `.pdf` vector):**
  1. **Truy hồi Top-5 đa miền (`retrieval_examples/`):**
     - Chọn 2 truy vấn trên **VeRi-776** (mặt đất) và **2–3 truy vấn trên VRAI / VRU** (góc nhìn từ trên cao của UAV).
     - Với mỗi truy vấn, vẽ **so sánh 2 hàng cạnh nhau**: Hàng trên là Top-5 của `ResNet50-IBN + RGA`, Hàng dưới là Top-5 của `ConvNeXt + BCGG` (ưu tiên chọn các ca mà `ResNet` xếp sai ở hạng 1 còn `ConvNeXt + BCGG` sửa đúng, kèm thêm **1 ca thất bại (Failure case)** mà cả hai cùng nhầm để phân tích hạn chế trong báo cáo).
  2. **Bản đồ nhiệt chú ý thật (`attention_heatmaps/`):**
     - Bật `set_attention_capture(True)` và `assert last_attention is not None`.
     - Vẽ 4 cột trên cùng 1 hàng: `Ảnh gốc` | `ResNet50-IBN + RGA (ga3)` | `ConvNeXt + RGA (ga3)` | `ConvNeXt + BCGG (ga3)` cho **2 ảnh VeRi-776** và **2 ảnh VRAI/VRU (góc nhìn UAV)**. Lưu kèm tệp `heatmap_stats.json` ghi rõ `min`, `max`, `mean`, `std` của bản đồ chú ý thô trước khi chuẩn hóa màu.
  3. **Biểu đồ số liệu (`charts/`):**
     - `delta_map_all_datasets.pdf`: Biểu đồ chênh lệch mAP khi đổi Backbone (`ConvNeXt+RGA - ResNet+RGA`) và khi đổi Cổng (`ConvNeXt+BCGG - ConvNeXt+RGA`) trên đủ 4 bộ dữ liệu (`VeRi-776`, `VehicleID S/M/L`, `VRU S/M/L`, `VRAI`).
     - `latency_fp32_fp16_trt.pdf`: Biểu đồ khoảng phân vị `p5–p50–p95` độ trễ mã hóa 1 ảnh ở `BS=1` so sánh `PyTorch FP32`, `PyTorch FP16`, `TensorRT FP16` và `End-to-End TRT FP16` kèm đường đứt nét mốc thời gian thực `33.33 ms (30 FPS)` và `40.00 ms (25 FPS)`.
     - `accuracy_vs_latency_pareto.pdf`: Biểu đồ đánh đổi (Pareto frontier) giữa `mAP (%)` trên trục tung và `Độ trễ p50 (ms)` trên trục hoành cho cả 3 mô hình.

## 3. Lệnh kiểm tra hoàn tất (Checklist cho AI trên Jetson trước khi báo xong)
Trước khi kết thúc, AI agent trên Jetson cần tự kiểm tra 5 điều kiện sau:
- [ ] Tất cả 12 checkpoint đều có mặt trong `00_env_and_manifest/checkpoints_manifest.csv` với đúng kiến trúc (`resnet50_ibn`, không phải `resnet50`).
- [ ] `01_accuracy_benchmarks/accuracy_summary.csv` có đủ kết quả của cả 4 bộ dữ liệu (`VeRi-776`, `VehicleID` 3 split × 10 lượt, `VRU` 3 split, `VRAI` kèm số query hợp lệ sau lọc).
- [ ] `02_bcgg_deploy_equivalence/bcgg_equivalence_report.json` có mã `SHA256` thật của checkpoint đã huấn luyện và sai số tensor ở FP32 nhỏ hơn `1e-4`.
- [ ] `03_efficiency_and_latency_orin/` có đủ bảng PyTorch (`FP32` & `FP16`), TensorRT (`FP16`) và End-to-End 1 ảnh cắt.
- [ ] `05_visualizations/attention_heatmaps/` có ảnh của cả VeRi-776 và VRAI/VRU (góc nhìn UAV) kèm `heatmap_stats.json` xác nhận kích hoạt thật từ hook.
