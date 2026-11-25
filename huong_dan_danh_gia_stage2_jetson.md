# ĐẶC TẢ HỆ THỐNG ĐÁNH GIÁ & BENCHMARK STAGE 2 (UAVREIDNET) TRÊN JETSON AGX ORIN

Tài liệu này là bản hướng dẫn kỹ thuật chi tiết để AI coding agent trên máy **NVIDIA Jetson AGX Orin** viết script và chạy toàn bộ thực nghiệm cho **Giai đoạn 2 (`UAVReIDNet`)** trên bộ dữ liệu **UAV-Anti-UAV**, phục vụ trực tiếp cho Chương 4 và Chương 6 của khóa luận.

## 0. Quy ước đầu vào & Nguyên tắc bắt buộc

### Các checkpoint đầu vào (đọc từ `eval_stage2_config.yaml`)
- **Mô hình chính:** `uavreid_convnext_bcgg_mamba` (`ConvNeXt + BCGG` từ Stage 1 + `Bidirectional Mamba` + `AttentionPooling` + `ReIDHead`).
- **Đối chứng bắt buộc 1 (Tách riêng tác dụng của Mamba):** `uavreid_convnext_bcgg_mean_head` (`ConvNeXt + BCGG` + `Visual Mean` + `ReIDHead/BNNeck`, không có Mamba).
- **Đối chứng bắt buộc 2 (So cổng RGA vs BCGG ở Stage 2):** `uavreid_convnext_rga_mamba` (`ConvNeXt + RGA` + `Bidirectional Mamba`).
- **Đối chứng mở rộng (nếu có train):** `uavreid_convnext_bcgg_attn` (`TemporalAttentionEncoder` / Transformer) và `uavreid_convnext_bcgg_mamba_meanpool` (`Mamba + MeanPooling`) (cái này chưa có).

### 5 Nguyên tắc kỹ thuật bắt buộc
1. **Dữ liệu sạch & có FPS thật:** Trước khi đánh giá, xác minh nguồn gốc mọi chuỗi trong tập đã xử lý (loại bỏ hoặc ghi rõ xuất xứ 127 chuỗi không mang tiền tố `UAV-Anti-UAV_`, đảm bảo Train và Test không trùng video/ID). Trích xuất `FPS` thực tế từ các file video gốc bằng `ffprobe`/`cv2` để quy đổi chính xác thời gian thu khung sang mili giây.
2. **Nạp checkpoint minh bạch 100%:** Khi nạp từng checkpoint, bắt buộc kiểm tra và khẳng định **100% khóa của bộ trích đặc trưng (`backbone.*`, `temporal_encoder.*`, `head.bnneck.*`) khớp hoàn toàn**. Chỉ cho phép bỏ qua `head.classifier.*` và phải in đích danh tên các khóa bị bỏ qua vào manifest.
3. **Chỉ dùng bộ độ đo truy hồi chuẩn (Không dùng Calibration Threshold / TAR@FAR):** Mọi bảng độ chính xác báo cáo đồng nhất **`mAP`, `Rank-1`, `Rank-5`, `Rank-10`, `mINP`** (theo `%`), tính cả **trung bình theo truy vấn (per-query)** và **trung bình theo video (per-video)**.
4. **Khóa xung Orin & Reset VRAM:** Chạy `sudo nvpmodel -m 0 && sudo jetson_clocks` trước khi đo độ trễ; luôn gọi `del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()` trước mỗi cấu hình.
5. **Xuất song song `.csv` và `.json`:** Mọi bảng số liệu xuất `.csv` chuẩn để đưa thẳng vào LaTeX/vẽ biểu đồ, kèm `.json` lưu metadata, seed và đường dẫn checkpoint (`SHA256`).

## 1. Cấu trúc thư mục đầu ra (`eval_results_stage2/`)

```text
eval_results_stage2/
├── 00_env_and_manifest/
├── 01_accuracy_benchmarks/
├── 02_occlusion_and_oov_analysis/
├── 03_efficiency_and_latency_orin/
├── 04_precision_consistency/
└── 05_visualizations/
```

## 2. Yêu cầu chi tiết cho từng thư mục

### Thư mục 1: `00_env_and_manifest/` — Kiểm kê Môi trường, Dữ liệu sạch & Checkpoint
- **Script đề xuất:** `00_audit_stage2.py`
- **Nhiệm vụ:**
  1. Ghi cấu hình phần cứng/phần mềm Jetson Orin (`nvpmodel`, CUDA, TensorRT, PyTorch, nhiệt độ GPU).
  2. Kiểm kê tập dữ liệu `UAV-Anti-UAV` đã xử lý: tổng số video Train/Test, tổng số cặp `(T_before, T_after)`, kiểm tra không giao nhau giữa Train và Test, thống kê phân bố độ dài khoảng vắng (trung bình, trung vị, min, max), số lượng clip bị thiếu khung phải đệm lặp (`m < L`), và bảng phân bố `FPS` thực tế đọc từ video gốc.
  3. Kiểm kê toàn bộ checkpoint Stage 2: `SHA256`, cấu hình thực tế (`backbone`, `gate`, `temporal_type`, `temporal_pool`, `use_pe`, `num_frames`, `stride`), trạng thái nạp khóa (`matched_feature_keys`, `ignored_classifier_keys`, `unexpected_keys`).
- **Tệp đầu ra bắt buộc:**
  - `env_info.json`
  - `dataset_audit.json` và `dataset_events_table.csv` (có thêm cột `fps`, `valid_before_frames`, `valid_after_frames`, `gap_frames`, `gap_seconds`)
  - `checkpoints_manifest.json` và `checkpoints_manifest.csv`

### Thư mục 2: `01_accuracy_benchmarks/` — Đánh giá Truy hồi Tổng thể & Kiểm định Thống kê
- **Script đề xuất:** `01_eval_uav_accuracy.py`
- **Các cấu hình bắt buộc đánh giá trên cùng tập Query và cùng Gallery:**
  1. `GASNet 1 khung` (khung cuối trước khi mất dấu làm gallery, khung đầu sau khi thấy lại làm query — chạy từ backbone `ConvNeXt + BCGG`).
  2. `GASNet trung bình L khung (không BNNeck đầu ReID)` (trung bình đặc trưng thị giác $L$ khung).
  3. `Đối chứng: Visual Mean + ReIDHead/BNNeck` (`Model B` — tách riêng hiệu ứng của `BNNeck`).
  4. `UAVReIDNet chính: ConvNeXt + BCGG + Bidirectional Mamba + AttentionPooling` (`Model A`).
  5. `Đối chứng cổng: ConvNeXt + RGA + Bidirectional Mamba` (`Model C`).
  6. *(Nếu có train)* `TemporalAttentionEncoder (Transformer)` và `Mamba + MeanPooling`.
  7. Đánh giá nhánh phụ của `Model A`: riêng `temporal_token` (512 chiều) và không gian `pre_bn` (1.472 chiều trước `BNNeck`).
- **Độ đo & Kiểm định bắt buộc:**
  - Tính đủ **`mAP`, `Rank-1`, `Rank-5`, `Rank-10`, `mINP`** theo 2 góc nhìn: **Per-Query** (trung bình trên toàn bộ truy vấn) và **Per-Video** (tính trung bình trong từng video rồi lấy trung bình đều trên số video test).
  - **Phân tích cặp đổi hạng (Paired Win/Loss):** Đếm chính xác số truy vấn mà `UAVReIDNet (Model A)` sửa đúng so với `Visual Mean + BNNeck (Model B)` và số truy vấn bị tụt hạng.
  - **Bootstrap theo video (10.000 lượt, `seed = 42`):** Tính khoảng tin cậy 95% (`[p2.5, p97.5]`) cho chênh lệch `Rank-1` và `mAP` giữa `Model A` với `GASNet 1 khung` và giữa `Model A` với `Model B`.
- **Tệp đầu ra bắt buộc:**
  - `uav_accuracy_summary.csv` (cột: `model, eval_mode, num_frames, per_query_mAP, per_query_R1, per_query_R5, per_query_R10, per_query_mINP, per_video_R1, per_video_mAP`)
  - `per_query_ranks.csv` (lưu `sequence_id, event_index, rank1_pos, ap, inp` cho từng mô hình để đối chiếu)
  - `bootstrap_ci_and_winloss.json`

### Thư mục 3: `02_occlusion_and_oov_analysis/` — Phân tích Chuyên sâu Che khuất, Mất tầm nhìn & Độ dài Chuỗi
- **Script đề xuất:** `02_eval_attributes_and_robustness.py`
- **4 phép phân tích bắt buộc (chạy trên cùng `Model A`, `Model B`, `GASNet 1 khung`):**
  1. **Theo nhóm thuộc tính không giao nhau (Disjoint Attribute Groups):** `Chỉ rời trường nhìn (OV)`, `Chỉ che khuất (OC)`, `Cả hai cờ (OV + OC)`, `Không có hai cờ` — báo cáo `Query count`, `mAP`, `Rank-1`, `mINP`.
  2. **Theo từng thuộc tính thách thức riêng lẻ của UAV-Anti-UAV:** `Out-of-View`, `Occlusion`, `Fast Motion`, `Scale Variation`, `Motion Blur`, `Low Resolution`, `Similar Distractors / Fake UAV`, `Background Clutter`.
  3. **Theo độ dài khoảng vắng (Gap Duration) & Số khung thực (Valid Frames):**
     - Chia khoảng vắng theo 4 nhóm (`1–4`, `5–16`, `17–64`, `≥65` khung; kèm quy đổi giây theo FPS thật).
     - Tách riêng nhóm **clip đủ khung** (`m = L`) và **clip ngắn bị đệm lặp** (`m < L`) để đánh giá hiệu quả của `AttentionPooling`.
  4. **Kiểm tra độ nhạy thứ tự thời gian & Độ dài cửa sổ $L$:**
     - Chạy đánh giá `Model A` tại **`L ∈ {4, 8, 12, 16}`** (báo cáo `mAP`, `Rank-1`, `mINP` và độ dao động `spread = max - min` để chứng minh tính ổn định theo $L$).
     - Chạy phép thử **hoán vị ngẫu nhiên thứ tự khung** (`shuffle`, cố định `seed = 42`) và **đảo ngược clip** (`reverse`) trên `Model A` để kiểm chứng mức độ phụ thuộc vào trình tự thời gian của Mamba.
- **Tệp đầu ra bắt buộc:**
  - `by_disjoint_oc_ov.csv`
  - `by_all_attributes.csv`
  - `by_gap_duration.csv`
  - `by_valid_frames_padding.csv`
  - `by_seq_len_and_order_ablation.csv`

### Thư mục 4: `03_efficiency_and_latency_orin/` — Đo Chi phí Tính toán & Độ trễ Video Thời gian thực trên Orin
- **Script đề xuất:** `03a_stage2_complexity.py`, `03b_bench_stage2_pytorch.py`, `03c_export_and_bench_stage2_trt.py`
- **Phần A — Độ phức tạp tĩnh (`stage2_static_complexity.csv`):**
  - Cho đầu vào 1 clip `1 × L × 3 × 224 × 224` (`L = 12` hoặc `16`), tách riêng: `backbone_params_M`, `temporal_head_params_M`, `total_params_M`, `backbone_GFLOPs`, `temporal_head_GFLOPs`, `total_GFLOPs`.
  - So sánh trực tiếp: `ConvNeXt+RGA+Mamba` vs `ConvNeXt+BCGG(folded)+Mamba` vs *(nếu có)* `ConvNeXt+BCGG(folded)+Transformer`.
- **Phần B — Độ trễ PyTorch trên Orin (`stage2_pytorch_latency.csv`):**
  - Đo ở `FP32` và `FP16` cho `BS = 1`, tại các độ dài chuỗi **`L ∈ {4, 8, 12, 16}`** (50 vòng warm-up, 200 vòng đo bằng `torch.cuda.Event`).
  - Ghi rõ: `backbone_clip_p50_ms`, `temporal_head_p50_ms`, `total_clip_p50_ms`, `total_clip_p95_ms`, `peak_vram_MiB`.
- **Phần C — Xuất ONNX & Đo tốc độ TensorRT FP16 (2 chế độ `kClip` và `kStream`):**
  - Xuất 2 file ONNX độc lập (dùng `BCGG` đã `to_deploy()`):
    1. `uav_backbone.onnx`: input `(B, 3, 224, 224)` $\rightarrow$ output `visual_feat (B, 960)`.
    2. `uav_temporal.onnx`: input `seq_feats (B, L, 960)` $\rightarrow$ output `reid_embed (B, 1472)` (đã L2-normalize).
  - Build 2 engine TensorRT **FP16** bằng `trtexec --fp16` và đo ở **2 kịch bản thực tế**:
    - **Kịch bản 1 — Offline / Batch Clip (`kClip`):** Đưa cả `L` khung qua `backbone.engine` (batch `L`) rồi qua `temporal.engine` (batch 1). Đo `p50_ms`, `p95_ms`, `clips_per_sec`.
    - **Kịch bản 2 — Online Streaming (`kStream` — sát thực tế bay UAV nhất):** Giả lập luồng camera gửi từng khung (mỗi bước chạy `backbone.engine` batch 1 rồi đẩy vào bộ đệm vòng `L` slot; khi đủ `L` khung chạy `temporal.engine` batch 1 + nhân cosine với thư viện 2.630 tracklet). Đo tách biệt:
      - `per_frame_backbone_p50_ms` (độ trễ mỗi khung hình khi đang thu thập).
      - `temporal_and_match_p50_ms` (độ trễ chốt định danh khi nhận khung thứ $L$).
      - **Ngân sách thời gian phản hồi từ lúc UAV tái xuất hiện:** Ghi bảng đối chiếu tách bạch `Thời gian chờ camera thu đủ L khung = (L - 1) * stride / FPS` (tính theo `L ∈ {4, 8, 12, 16}`) và `Thời gian xử lý trên Orin (ms)`.
- **Tệp đầu ra bắt buộc:**
  - `stage2_static_complexity.csv`
  - `stage2_pytorch_latency.csv`
  - `stage2_tensorrt_latency.csv`
  - `stage2_reacquisition_time_budget.csv`

### Thư mục 5: `04_precision_consistency/` — Kiểm chứng Sai số giữa PyTorch FP32, FP16 và TensorRT FP16
- **Script đề xuất:** `04_verify_stage2_trt_accuracy.py`
- **Nhiệm vụ:**
  - Chạy trích xuất đặc trưng trên tập Test `UAV-Anti-UAV` bằng cả 3 runtime: `PyTorch FP32`, `PyTorch FP16`, và `TensorRT FP16` cho `Model A`.
  - Đo `min_cosine_sim` và `mean_abs_diff` giữa embedding PyTorch FP32 và TensorRT FP16, đồng thời so sánh trực tiếp `mAP`, `Rank-1`, `Rank-5`, `mINP` để chứng minh tăng tốc TensorRT FP16 không làm giảm chất lượng tái nhận diện UAV.
- **Tệp đầu ra bắt buộc:**
  - `stage2_precision_comparison.csv`
  - `trt_vs_pytorch_embedding_diff.json`

### Thư mục 6: `05_visualizations/` — Hình ảnh Trực quan hóa & Biểu đồ cho Chương 6 (300 DPI + PDF)
- **Script đề xuất:** `05_generate_stage2_visuals.py`
- **Các nhóm hình cần xuất (lưu cả `.png` 300 DPI và `.pdf` vector):**
  1. **Ví dụ truy hồi UAV sau che khuất / mất tầm nhìn (`uav_retrieval_examples/`):**
     - Chọn **2 ca thành công tiêu biểu** (1 ca `Out-of-View` khoảng vắng dài, 1 ca `Occlusion`) nơi `GASNet 1 khung` hoặc `Visual Mean` xếp sai còn `UAVReIDNet (Model A)` đưa đúng UAV lên hạng 1; kèm **1 ca thất bại (Failure case)** (ví dụ UAV cực nhỏ `<15px` hoặc bị nhầm với `Fake UAV`) để phân tích hạn chế.
     - Bố cục mỗi hình: Hiển thị 3 khung đại diện của chuỗi `Query (T_after)` ở bên trái, bên phải hiển thị Top-5 ứng viên `Gallery (T_before)` (mỗi ứng viên hiển thị khung đại diện + khung viền xanh ĐÚNG / đỏ SAI + điểm cosine + độ dài khoảng vắng).
  2. **Trực quan hóa trọng số `AttentionPooling` theo thời gian (`temporal_weights/`):**
     - Vẽ 2–3 dải clip 12/16 khung hình kèm thanh trọng số $\alpha_t$ của `AttentionPooling` bên dưới mỗi khung, minh họa mô hình tự động giảm trọng số các khung bị mờ/bị đệm lặp và dồn trọng số vào các khung rõ nét nhất.
  3. **Biểu đồ phân tích định lượng (`charts/`):**
     - `rank1_by_gap_duration.pdf`: Biểu đồ đường/cột so sánh `Rank-1` của `GASNet 1 khung`, `Visual Mean + BNNeck` và `UAVReIDNet (Mamba)` qua 4 mức khoảng vắng (`1–4`, `5–16`, `17–64`, `≥65` khung).
     - `rank1_by_oc_ov_attributes.pdf`: Biểu đồ cột nhóm so sánh `Rank-1` trên 4 nhóm thuộc tính (`Chỉ OV`, `Chỉ OC`, `Cả hai`, `Không có`).
     - `seq_len_stability.pdf`: Biểu đồ `Rank-1` và `mAP` theo độ dài chuỗi $L \in \{4, 8, 12, 16\}$ kèm thời gian chờ thu khung.
     - `stage2_latency_breakdown.pdf`: Biểu đồ cột chồng so sánh độ trễ xử lý clip trên Orin giữa `PyTorch FP32`, `PyTorch FP16` và `TensorRT FP16` (`kClip` và `kStream`).

## 3. Checklist tự kiểm tra trước khi bàn giao kết quả Stage 2
- [ ] `00_env_and_manifest/dataset_audit.json` xác nhận dữ liệu sạch (không còn chuỗi không rõ nguồn gốc, có FPS thật từ video gốc).
- [ ] `00_env_and_manifest/checkpoints_manifest.csv` xác nhận 100% khóa đặc trưng của mọi checkpoint khớp sạch (không còn lỗi bí ẩn kiểu `888/891`).
- [ ] `01_accuracy_benchmarks/uav_accuracy_summary.csv` có đủ kết quả của `GASNet 1 khung`, `Visual Mean + BNNeck` (đối chứng tách Mamba), `UAVReIDNet (ConvNeXt + BCGG + Mamba)` và `ConvNeXt + RGA + Mamba`.
- [ ] `03_efficiency_and_latency_orin/stage2_tensorrt_latency.csv` có đủ số đo độ trễ thật trên Orin cho cả `kClip` và `kStream` (không còn mục `skipped`).
- [ ] `05_visualizations/uav_retrieval_examples/` có đủ hình minh họa truy hồi UAV thành công và thất bại ở 300 DPI.
