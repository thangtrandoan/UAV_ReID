# Hướng Dẫn Bộ Công Cụ Đánh Giá Thực Nghiệm GASNet Giai Đoạn 1 (Stage 1)

Bộ công cụ này cung cấp toàn bộ mã nguồn phục vụ việc chạy thực nghiệm, đánh giá các độ đo chính xác (CMC, mAP, mINP), hiệu năng tính toán (#Params, GFLOPs, Latency, FPS, Peak VRAM) và trực quan hóa (Attention Heatmaps, Top-5 Retrieval) để đưa vào Báo cáo luận văn / bài báo.

---

## 1. Cấu trúc thư mục

```text
scripts/eval_gasnet/
├── __init__.py               # Khởi tạo package
├── metrics.py                # Tính CMC (Rank-1, 5, 10), mAP, mINP, lọc junk đa camera
├── datasets.py               # Tải VeRi-776, VRU / VehicleID (Small/Medium/Large), tự nhận diện /data hoặc ./data
├── efficiency.py             # Đo #Params, GFLOPs, Latency (50 warmup + 200 reps), FPS, Peak VRAM
├── models.py                 # Nạp các biến thể GASNet từ weights/ (ResNet+RGA, ConvNeXt+RGA, ConvNeXt+BCGG Train/Deploy)
├── gate_benchmark.py         # Thí nghiệm Bảng 5.3: Đo riêng khối Gate (RGA vs BCGG) ở các độ phân giải 224, 256, 384
├── visualize.py              # Trích xuất last_attention tạo Attention Heatmaps và sinh ảnh truy vấn Top-5
└── run_all.py                # Script chạy tự động tổng hợp toàn bộ thí nghiệm và xuất bảng Markdown/CSV
```

---

## 2. Các lệnh chạy nhanh

### A. Chạy toàn bộ tất cả thí nghiệm và sinh ảnh trực quan
```bash
python3 -m scripts.eval_gasnet.run_all --all
```
*Kết quả bảng và ảnh trực quan sẽ tự động xuất ra thư mục `outputs/gasnet_report/`.*

### B. Chạy riêng từng bảng thí nghiệm

#### 1. Thí nghiệm 1 — Bảng 5.1: So sánh Backbone (ResNet50-IBN vs DINO-ConvNeXt)
```bash
python3 -m scripts.eval_gasnet.run_all --table 5.1
```
* So sánh 3 cấu hình:
  * `ResNet50-IBN (RGA)`: Baseline backbone giữ nguyên khối RGA.
  * `DINO-ConvNeXt (RGA)`: Đổi sang backbone DINO-ConvNeXt và giữ nguyên RGA (cô lập chính xác hiệu quả của việc thay thế backbone).
  * `DINO-ConvNeXt (BCGG)`: Phiên bản nâng cấp toàn diện cả backbone lẫn khối gate BCGG.
* Đánh giá độc lập trên từng tập dữ liệu riêng biệt:
  * **VeRi-776**: 1,678 queries, 11,579 gallery.
  * **VehicleID**: 1,232 queries (Cam A), 1,232 gallery (Cam B) từ `data/VehicleID/img_database`.
  * **VRU**: Đánh giá trên 3 split riêng biệt:
    * `Small (1200)`: 12,720 queries, 1,200 gallery.
    * `Medium (2400)`: 24,945 queries, 2,400 gallery.
    * `Large (8000)`: 83,595 queries, 8,000 gallery.
  * **VRAI**: Tập kiểm thử chuẩn UAV góc nhìn trên cao (1,000 identities validation).
* Độ đo tính toán: mAP, Rank-1, Rank-5, Rank-10, mINP.

#### 2. Thí nghiệm 2 — Bảng 5.2: Ablation Lũy tiến của GASNet
```bash
python3 -m scripts.eval_gasnet.run_all --table 5.2 --ablation-dataset vru
```
* Tùy chọn tập dữ liệu đánh giá độ chính xác qua `--ablation-dataset`: `vru` (mặc định VRU Small), `veri` (VeRi-776), `vehicleid`, hoặc `vrai`.
* Đánh giá lũy tiến 4 cấu hình:
  * Hàng 1: GASNet gốc (`ResNet-50 + RGA`)
  * Hàng 2: GASNet + `DINO-ConvNeXt + RGA`
  * Hàng 3: GASNet + `DINO-ConvNeXt + BCGG` (Train mode)
  * Hàng 4: GASNet + `DINO-ConvNeXt + BCGG` (Deploy mode - RepDW Folded)
* Đo đầy đủ hệ thống độ đo:
  * **Tham số**: #Params Tổng (M) và riêng khối Gate #Params (K) (chứng minh BCGG cắt giảm 99% tham số gate).
  * **Chi phí tính toán**: GFLOPs và GMACs (trên tensor 1×3×224×224).
  * **Độ trễ suy luận**: Tách riêng Latency Tổng (ms), Backbone Time (ms) và Gate + Head Time (ms).
  * **Thông lượng**: Throughput trực tuyến (BS=1 FPS) và theo lô (Batch FPS).
  * **Bộ nhớ**: Peak GPU VRAM (MB) qua `torch.cuda.max_memory_allocated()`.
  * **Độ chính xác**: mAP, Rank-1, Rank-5, Rank-10, mINP.

#### 3. Thí nghiệm 2 — Bảng 5.3: So sánh Chi phí Tính toán riêng khối Gate (RGA vs BCGG)
```bash
python3 -m scripts.eval_gasnet.run_all --table 5.3
```
* So sánh ở các độ phân giải:
  * 224×224 (Feature size: 14×14)
  * 256×256 (Feature size: 16×16)
  * 384×384 (Feature size: 24×24)
* Đo các chỉ số: #Params (K), GFLOPs, Latency (ms), Peak VRAM (MB) và Đặc tính / Khả năng mở rộng.
* Làm nổi bật giới hạn lỗi kích thước cố định của RGA và bùng nổ bộ nhớ bậc hai O((HW)^2) so với tính thích ứng và tuyến tính O(HW) của BCGG.

#### 4. Trực quan hóa (Qualitative Analysis)
```bash
python3 -m scripts.eval_gasnet.run_all --visualize --vis-dataset veri --num-vis 5
```
* Tùy chọn tập dữ liệu trực quan qua `--vis-dataset`: `veri`, `vru`, `vehicleid`, `vrai`.
* Sinh bản đồ nhiệt chú ý Attention Heatmaps (RGA vs BCGG) từ `last_attention` so sánh độ tập trung vào thân xe và loại bỏ nền.
* Sinh bảng ảnh truy vấn Top-5 (Query + Top-5 Gallery) với viền Xanh lá cho ảnh đúng (True Match) và viền Đỏ cho ảnh sai (False Match).

---

## 3. Tùy chỉnh tham số đầu vào

* `--data-root /path/to/data`: Chỉ định thư mục dữ liệu nếu không nằm ở `/data` hoặc `./data`.
* `--weights-dir weights`: Chỉ định thư mục chứa các file trọng số `.pth`.
* `--batch-size 64`: Kích thước batch khi trích xuất đặc trưng (mặc định 64).
* `--device cuda`: Thiết bị chạy (`cuda` hoặc `cpu`).
* `--output-dir outputs/gasnet_report`: Thư mục lưu kết quả.

---

## 4. Định dạng kết quả xuất ra

Sau khi chạy xong, thư mục `outputs/gasnet_report/` sẽ có cấu trúc:
```text
outputs/gasnet_report/
├── table_5_1_backbone_comparison.md   # Bảng 5.1 định dạng Markdown
├── table_5_1_backbone_comparison.csv  # Bảng 5.1 định dạng CSV
├── table_5_2_ablation_gasnet.md       # Bảng 5.2 định dạng Markdown
├── table_5_2_ablation_gasnet.csv      # Bảng 5.2 định dạng CSV
├── table_5_3_gate_comparison.md       # Bảng 5.3 định dạng Markdown
├── table_5_3_gate_comparison.csv      # Bảng 5.3 định dạng CSV
└── visualizations/
    ├── attention_heatmaps/            # Ảnh so sánh bản đồ nhiệt [Gốc | RGA | BCGG]
    └── retrieval_top5/                # Ảnh bảng truy vấn Top-5 có viền Xanh/Đỏ
```
Các bảng Markdown và CSV này có thể copy trực tiếp vào báo cáo luận văn hoặc bảng biểu khoa học.
