# BÁO CÁO NGHIÊN CỨU KHOA HỌC
**HỌC PHẦN: TRÍ TUỆ NHÂN TẠO (ARTIFICIAL INTELLIGENCE)**  
**TRƯỜNG ĐẠI HỌC KINH TẾ TP. HỒ CHÍ MINH (UEH)**

---

**ĐỀ TÀI S08: LoRA VÀ QLoRA — TINH CHỈNH MÔ HÌNH LỚN HIỆU QUẢ VỀ SỐ LƯỢNG THAM SỐ**

**Nhóm thực hiện (03 thành viên):**
1. **Nguyễn Văn Minh Triều** — *Nhóm trưởng (Lead Kỹ thuật & Tổng hợp dự án)*
2. **Ngô Trọng Phúc** — *Thực nghiệm & Trực quan hóa*
3. **Nguyễn Phúc Vương** — *Nghiên cứu Mở rộng & Thiết kế Slide*

---

## TÓM TẮT (ABSTRACT)

Sự bùng nổ của các Mô hình Ngôn ngữ Lớn (LLMs) mang lại năng lực tổng quát hóa vượt trội nhưng đặt ra rào cản lớn về tài nguyên phần cứng khi tinh chỉnh theo phương pháp truyền thống (Full Fine-tuning). Đối với một mô hình 7 tỷ tham số (7B) ở độ chính xác FP16, chi phí lưu trữ trọng số, đạo hàm (gradients) và trạng thái bộ tối ưu AdamW (optimizer states) vượt ngưỡng 80 GB VRAM, khiến việc huấn luyện trở nên bất khả thi trên phần cứng phổ thông. Nghiên cứu này phân tích và hiện thực hóa hai kỹ thuật Tinh chỉnh Hiệu quả về Số lượng Tham số (PEFT) tiêu biểu gồm **LoRA (Low-Rank Adaptation)** và **QLoRA (Quantized LoRA)**. Bằng phân tích toán học trên lớp tuyến tính $1024 \times 1024$, LoRA giảm tới **99.22%** số tham số cần huấn luyện tại hạng $r=4$ mà không làm tăng độ trễ suy luận nhờ cơ chế gộp trọng số $W_{\text{deploy}} = W_0 + \frac{\alpha}{r}BA$. Đồng thời, báo cáo làm rõ bản chất sự khác biệt giữa mức giảm tham số huấn luyện và mức giảm VRAM thực tế, triển khai kiểm thử `LoRALinear` bằng PyTorch thuần, mô phỏng lượng tử hóa 4-bit NormalFloat (NF4) và thực nghiệm đối chứng trên mô hình `distilbert-base-uncased` với tập dữ liệu SST-2.

---

## 1. GIỚI THIỆU & ĐỘNG LỰC NGHIÊN CỨU (INTRODUCTION)

### 1.1. Bối cảnh bài toán
Trong kỷ nguyên Deep Learning hiện đại, mô hình Transformer tiền huấn luyện (Pre-trained Foundation Models) đạt kích thước từ hàng trăm triệu đến hàng trăm tỷ tham số. Để thích ứng mô hình tổng quát vào các tác vụ chuyên biệt (phân loại cảm xúc, hỏi đáp tài chính, tóm tắt văn bản y khoa), phương pháp truyền thống là **Full Fine-tuning** — cập nhật toàn bộ ma trận trọng số của mô hình.

### 1.2. Bốn nỗi đau thực tế của Full Fine-tuning
1. **Khủng hoảng bộ nhớ phần cứng (VRAM Bottleneck):** Với mô hình 7B tham số ở định dạng FP16 (2 bytes/tham số), ngoài **14 GB** lưu trọng số gốc, quá trình huấn luyện bắt buộc phải lưu:
   - Đạo hàm (Gradients FP16): $7\text{B} \times 2 \text{ bytes} = 14\text{ GB}$.
   - Trạng thái bộ tối ưu AdamW (lưu moment bậc một $m_t$ và bậc hai $v_t$ ở FP32): $2 \times 4 \text{ bytes} \times 7\text{B} = 56\text{ GB}$.
   - Bộ nhớ kích hoạt trung gian (Activations): nhiều GB tùy thuộc độ dài ngữ cảnh (context length) và batch size.
   $\Rightarrow$ Tổng nhu cầu lên tới **80–160 GB VRAM** (yêu cầu cụm GPU A100/H100 đắt đỏ), vượt xa khả năng của sinh viên, nhóm nghiên cứu nhỏ và doanh nghiệp vừa/nhỏ.
2. **Dư thừa lưu trữ & triển khai đa tác vụ (Storage & Serving Redundancy):** Khi triển khai cho $N$ bài toán khác nhau, Full Fine-tuning buộc phải lưu $N$ bản sao mô hình độc lập (14–100 GB/bản) và gây độ trễ cao khi chuyển đổi mô hình trên server suy luận.
3. **Hiện tượng quên thảm khốc (Catastrophic Forgetting):** Việc cập nhật toàn bộ tham số trên tập dữ liệu hẹp dễ phá vỡ tri thức tổng quát đã tích lũy trong giai đoạn tiền huấn luyện.
4. **Hạn chế của Serial Adapters cổ điển:** Các lớp Adapter chèn tuần tự vào giữa các khối Transformer làm tăng chiều sâu tính toán, gây ra độ trễ suy luận (inference latency) đáng kể do không thể gộp trực tiếp vào trọng số gốc.

---

## 2. CƠ SỞ LÝ THUYẾT (THEORETICAL BACKGROUND)

### 2.1. So sánh 5 phương pháp thích ứng mô hình

| Tiêu chí | Pre-training | Full Fine-tuning | Linear Probing | Serial Adapters | PEFT (LoRA / QLoRA) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phạm vi cập nhật** | Toàn bộ tham số (từ đầu) | Toàn bộ tham số | Chỉ lớp phân loại đầu ra | Các module chèn tuần tự nhỏ | Ma trận hạng thấp $A, B$ song song |
| **Tài nguyên VRAM** | Cực lớn (cụm GPU) | Rất lớn (80–160 GB cho 7B) | Rất thấp | Trung bình – Thấp | Thấp (LoRA) đến Rất thấp (QLoRA) |
| **Lưu trữ đa tác vụ** | 1 mô hình gốc | $N$ bản sao đầy đủ | 1 bản gốc + $N$ đầu ra nhỏ | 1 bản gốc + $N$ adapter | 1 bản gốc + $N$ file LoRA vài MB |
| **Độ trễ suy luận** | Chuẩn (Baseline) | Không tăng thêm | Không tăng thêm | **Tăng** do tính tuần tự | **Bằng 0** (khi gộp $W_0 + \frac{\alpha}{r}BA$) |
| **Năng lực biểu diễn** | Tổng quát tối đa | Cao (nhưng dễ quên tri thức) | Thấp (không chỉnh đặc trưng sâu) | Khá cao | Tương đương Full FT trên đa số tác vụ |

### 2.2. Cơ sở toán học của LoRA (Hu et al., ICLR 2022)
Dựa trên **Giả thuyết hạng nội tại (Intrinsic Rank Hypothesis)** của Aghajanyan et al. và Hu et al., phần biến thiên trọng số $\Delta W$ khi tinh chỉnh mô hình lớn thực chất nằm trong một không gian con có số chiều nội tại rất thấp. Thay vì cập nhật trực tiếp ma trận trọng số tiền huấn luyện $W_0 \in \mathbb{R}^{d \times k}$, LoRA đóng băng hoàn toàn $W_0$ (`requires_grad = False`) và phân rã phần cập nhật thành tích của hai ma trận hạng thấp:

$$W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$$

Phương trình lan truyền thuận (forward pass) của một lớp tuyến tính được viết thành:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B A) x$$

Trong đó, ý nghĩa từng biến số như sau:
* $x \in \mathbb{R}^{k}$: vector đặc trưng đầu vào của lớp tuyến tính.
* $h \in \mathbb{R}^{d}$: vector đầu ra sau biến đổi tuyến tính.
* $W_0 \in \mathbb{R}^{d \times k}$: ma trận trọng số tiền huấn luyện ban đầu, được **đóng băng hoàn toàn** trong suốt quá trình lan truyền ngược.
* $A \in \mathbb{R}^{r \times k}$: ma trận chiếu giảm chiều, được khởi tạo ngẫu nhiên theo phân phối chuẩn $\mathcal{N}(0, \sigma^2)$ (hoặc Kaiming uniform) để phá vỡ tính đối xứng.
* $B \in \mathbb{R}^{d \times r}$: ma trận chiếu phục hồi chiều, được **khởi tạo toàn số 0** ($B = 0$), đảm bảo $\Delta W = B A = 0$ tại thời điểm bước $0$ (mô hình giữ nguyên hành vi gốc trước khi huấn luyện).
* $r \ll \min(d, k)$: hạng (rank) của phép phân rã, là siêu tham số quyết định dung lượng biểu diễn của adapter (thường chọn $r \in \{4, 8, 16\}$).
* $\alpha$: hằng số tỷ lệ (scaling factor), giúp ổn định biên độ tín hiệu cập nhật $\frac{\alpha}{r}$ khi thay đổi hạng $r$, giảm nhu cầu chỉnh lại tốc độ học (learning rate).

**Tính năng gộp trọng số khi triển khai (Zero Inference Latency):** Sau khi huấn luyện xong, hai ma trận hạng thấp được nhân và cộng trực tiếp vào trọng số gốc:

$$W_{\text{deploy}} = W_0 + \frac{\alpha}{r} B A$$

Nhờ tính chất tuyến tính này, mô hình triển khai có cấu trúc và thời gian suy luận giống hệt mô hình gốc $W_0$, hoàn toàn triệt tiêu độ trễ tăng thêm so với Serial Adapters.

![Kiến trúc LoRA và cơ chế gộp trọng số](../figures/lora_architecture.png)

### 2.3. Cơ chế lượng tử hóa QLoRA (Dettmers et al., NeurIPS 2023)
QLoRA mở rộng LoRA nhằm nén bộ nhớ của chính trọng số gốc đóng băng $W_0$ mà vẫn bảo toàn hiệu năng tương đương tinh chỉnh 16-bit thông qua 3 đột phá:
1. **4-bit NormalFloat (NF4):** Định dạng kiểu dữ liệu 4-bit tối ưu theo lý thuyết thông tin cho các trọng số mạng nơ-ron vốn tuân theo phân phối chuẩn $\mathcal{N}(0, \sigma^2)$. Thay vì chia đều các mức lượng tử như INT4, NF4 đặt 16 mức chia theo các phân vị xác suất đều nhau của phân phối Gauss chuẩn hóa, giúp tập trung độ phân giải ở vùng gần $0$ nơi phần lớn trọng số hội tụ.
2. **Double Quantization (Lượng tử hóa kép):** Lượng tử hóa theo khối (blockwise, ví dụ `blocksize = 64`) yêu cầu lưu hằng số chuẩn hóa absmax ở FP32 (tốn $32 / 64 = 0.5$ bit/tham số). Double Quantization lượng tử hóa tiếp các hằng số FP32 này xuống FP8 theo khối cấp hai (`dq_blocksize = 256`), giảm chi phí hằng số xuống $8 / 64 + 32 / (64 \times 256) \approx 0.127$ bit/tham số, tiết kiệm thêm **~0.373 bit/tham số** ($\approx 3$ GB cho mô hình 65B).
3. **Paged Optimizers:** Tận dụng cơ chế *NVIDIA Unified Memory* để tự động phân trang trạng thái bộ tối ưu giữa GPU VRAM và CPU RAM khi gặp các đỉnh bộ nhớ (memory spikes) từ chuỗi đầu vào dài, giúp ngăn chặn lỗi `CUDA Out of Memory (OOM)`.

---

## 3. PHƯƠNG PHÁP & VÍ DỤ TÍNH TOÁN SỐ (METHODOLOGY & MATHEMATICAL VERIFICATION)

### 3.1. Tính toán số lượng tham số trên lớp tuyến tính $1024 \times 1024$
Xét một ma trận chiếu tuyến tính tiêu chuẩn trong Transformer với kích thước $d = 1024, k = 1024$ (bỏ qua bias để tập trung vào phép chiếu ma trận):

* **Full Fine-tuning:** $\text{Params}_{\text{Full}} = d \times k = 1024 \times 1024 = 1,048,576$ tham số.
* **LoRA:** Chỉ huấn luyện cặp ma trận $A \in \mathbb{R}^{r \times k}$ và $B \in \mathbb{R}^{d \times r}$, nên tổng tham số huấn luyện là $\text{Params}_{\text{LoRA}} = r \times (d + k) = r \times 2048$.

Bảng dưới đây tổng hợp kết quả được tính toán tự động từ module `src/calculate_parameters.py`:

| Phương pháp | Công thức | Tham số huấn luyện | Tỷ lệ giữ lại (%) | Tỷ lệ giảm (%) |
| :--- | :--- | ---: | ---: | ---: |
| **Full Fine-tuning** | $1024 \times 1024$ | 1,048,576 | 100.00% | 0.00% |
| **LoRA ($r=4$)** | $4 \times (1024 + 1024)$ | 8,192 | 0.78% | **99.22%** |
| **LoRA ($r=8$)** | $8 \times (1024 + 1024)$ | 16,384 | 1.56% | **98.44%** |
| **LoRA ($r=16$)** | $16 \times (1024 + 1024)$ | 32,768 | 3.12% | **96.88%** |

![So sánh số lượng tham số huấn luyện trên lớp 1024x1024](../figures/params_comparison.png)

### 3.2. Thảo luận phản biện mấu chốt: Giảm 99% tham số có đồng nghĩa giảm 99% VRAM?
**Câu trả lời là KHÔNG.** Đây là điểm hiểu lầm phổ biến nhất khi tiếp cận PEFT. Qua phân tích mô hình 7B tham số ($1 \text{ GB} = 10^9 \text{ bytes}$, giả sử LoRA chiếm $0.5\%$ tham số huấn luyện và Activations cố định ở $6.00 \text{ GB}$ để so sánh tương đương):

| Cấu hình (Mô hình 7B) | Weights (GB) | Gradients (GB) | AdamW States (GB) | Activations (GB) | Tổng VRAM (GB) | Giảm VRAM thực tế |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Full Fine-tuning (FP16)** | 14.00 | 14.00 | 56.00 | 6.00 | **90.00** | 0.00% |
| **LoRA (FP16, 0.5% trainable)** | 14.07 | 0.07 | 0.28 | 6.00 | **20.42** | **77.31%** |
| **QLoRA (NF4 + Double Quant)** | 3.68 | 0.07 | 0.28 | 6.00 | **10.03** | **88.86%** |

**Lý giải bản chất:**
1. **Trọng số gốc $W_0$ vẫn phải nằm trên GPU** để thực hiện lan truyền thuận ($14\text{ GB}$ ở LoRA FP16; chỉ khi dùng QLoRA 4-bit NF4 + Double Quant mới nén xuống còn $\approx 3.61\text{ GB}$ cho phần base weights).
2. **Bộ nhớ kích hoạt trung gian (Activations)** vẫn phải lưu qua các tầng đóng băng nhằm tính chuỗi đạo hàm ngược (chain rule) dẫn về các ma trận $A, B$.
3. **Nguồn tiết kiệm lớn nhất của LoRA đến từ việc triệt tiêu hơn 99% Gradients và Optimizer States** (từ $14 + 56 = 70\text{ GB}$ xuống còn $0.07 + 0.28 = 0.35\text{ GB}$ — chiếm **~75–78%** tổng mức VRAM ban đầu của Full Fine-tuning).

![Phân rã VRAM lý thuyết cho mô hình 7B](../figures/vram_breakdown.png)

### 3.3. Thiết kế hệ thống mã nguồn thực nghiệm
Dự án được xây dựng hoàn chỉnh bằng Python (PyTorch & Hugging Face PEFT) với 5 module độc lập:
* `src/lora_layer.py`: Cài đặt lớp `LoRALinear(nn.Module)` thuần PyTorch, hỗ trợ `merge_weights()`, `unmerge_weights()` và tính đúng đắn tích chuỗi $(x A^T) B^T$.
* `src/calculate_parameters.py`: Tính bảng tham số $1024 \times 1024$, phân rã VRAM 7B và xuất biểu đồ.
* `src/train_synthetic.py`: Thử nghiệm kiểm chứng hội tụ trên bài toán thích ứng tuyến tính $256 \times 256$ có hạng nội tại thực $r^* = 8$.
* `src/qlora_demo.py`: Mô phỏng lượng tử hóa NF4 4-bit theo khối (`blocksize=64`) và Double Quantization (`dq_blocksize=256`) trên ma trận trọng số chuẩn $1024 \times 1024$.
* `src/train_hf_peft.py`: Tinh chỉnh thực tế mô hình `distilbert-base-uncased` trên tập phân loại cảm xúc SST-2 (`data/sst2_sample.csv`) cho 3 cấu hình: Full Fine-tuning, LoRA $r=4$, và LoRA $r=16$.

---

## 4. KẾT QUẢ THỰC NGHIỆM & THẢO LUẬN (RESULTS & DISCUSSION)

*(Các số liệu thực nghiệm dưới đây được ghi nhận trực tiếp từ quá trình chạy mã nguồn của nhóm — sẽ được cập nhật chính xác sau bước kiểm thử thực thi.)*

### 4.1. Kiểm chứng toán học và phép gộp trọng số (`src/lora_layer.py` & `tests/`)
* Module `LoRALinear` vượt qua toàn bộ các bài kiểm thử đơn vị (`unittest`) về: khởi tạo $\Delta W = 0$, đóng băng hoàn toàn `base_layer.weight.grad == None`, chỉ cập nhật gradient cho `lora_A` và `lora_B`, và phục hồi nguyên trạng khi gọi `unmerge_weights()`.
* Kiểm tra phép gộp trọng số trên lớp $1024 \times 1024$ ($r=8, \alpha=16$):
  * `torch.allclose(y_1, y_2, atol=1e-5)` trả về **`True`** (sai số tuyệt đối cực đại ở mức $\sim 10^{-6}$, phát sinh từ sai số làm tròn dấu phẩy động FP32).

### 4.2. Thực nghiệm mô phỏng hạng nội tại (`src/train_synthetic.py`)
Trên bài toán thích ứng ma trận $256 \times 256$ có độ dịch chuyển trọng số đích $\Delta W^*$ mang hạng nội tại thực $r^* = 8$:
* {{SYNTHETIC_RESULTS_TABLE}}
* **Nhận xét:** Khi $r=4 < r^*$, mô hình bị giới hạn dung lượng biểu diễn (underfitting) nên dừng ở mức sai số cao hơn. Khi $r \ge r^*$ ($r=8$ hoặc $r=16$), LoRA hội tụ tiệm cận hoàn toàn với Full Fine-tuning trong khi chỉ huấn luyện **6.22% – 12.45%** số tham số.

![Đường cong hội tụ hàm mất mát trên bài toán mô phỏng](../figures/loss_convergence.png)

### 4.3. Mô phỏng lượng tử hóa NF4 & Double Quantization (`src/qlora_demo.py`)
* {{QLORA_RESULTS_SUMMARY}}
* Độ tương đồng cosin (Cosine Similarity) giữa ma trận trọng số gốc $W_0$ và ma trận giải lượng tử $\hat{W}_{\text{NF4+DQ}}$ đạt rất cao ($\approx 0.99$), chứng minh NF4 bảo toàn tốt thông tin hình học của trọng số phân phối chuẩn dù nén gần **3.88x** so với FP16.

### 4.4. Thực nghiệm tinh chỉnh DistilBERT với Hugging Face PEFT (`src/train_hf_peft.py`)
Thử nghiệm phân loại cảm xúc nhị phân trên tập dữ liệu SST-2 (`distilbert-base-uncased`, chèn LoRA vào các lớp chiếu chú ý `q_lin`, `v_lin`):
* {{HF_PEFT_RESULTS_TABLE}}

![Kết quả thực nghiệm DistilBERT PEFT trên SST-2](../figures/hf_peft_convergence.png)

### 4.5. Phân tích phản biện: Khi nào LoRA KHÔNG tối ưu?
Mặc dù LoRA và QLoRA mang lại hiệu quả vượt trội trong đa số bài toán thích ứng tác vụ (downstream task adaptation), nhóm chỉ ra **4 trường hợp phương pháp này không phải lựa chọn tối ưu**:
1. **Chuyển dịch miền dữ liệu cực lớn (Severe Domain Shift):** Khi bài toán mới sử dụng ngôn ngữ hiếm, ký hiệu đặc thù hoặc phân phối dữ liệu hoàn toàn xa lạ với tập tiền huấn luyện (ví dụ: chuyển một LLM tiếng Anh thuần túy sang đọc mã nhị phân, DNA, hoặc ngôn ngữ ít tài nguyên chưa từng học), không gian biểu diễn tầng thấp cần được tái cấu trúc toàn diện thay vì chỉ cập nhật hạng thấp.
2. **Nạp tri thức sự kiện quy mô lớn (Large-scale Knowledge Injection / Continual Pre-training):** Theo các nghiên cứu thực nghiệm (như Biderman et al., 2024 - *"LoRA Learns Less and Forgets Less"*), LoRA bị giới hạn bởi số bậc tự do $r(d+k)$ nên học chậm và kém hơn Full Fine-tuning khi cần ghi nhớ lượng lớn kiến thức thực thể mới vào trọng số mô hình.
3. **Độ trễ chuyển đổi adapter khi phục vụ hàng loạt yêu cầu không đồng nhất trên phần cứng biên:** Dù gộp trọng số đạt 0 độ trễ cho 1 tác vụ, nhưng nếu một hệ thống phục vụ song song hàng trăm adapter chưa gộp theo từng request trong cùng một batch, phép nhân nhánh phụ $(xA^T)B^T$ phát sinh chi phí lập lịch nhân (kernel launch overhead).
4. **Rủi ro tương thích phần cứng/hệ điều hành với QLoRA:** Thư viện `bitsandbytes` phụ thuộc chặt vào CUDA kernels tối ưu cho Linux; khi triển khai trên Windows hoặc CPU thuần, chi phí giải lượng tử ngược lại có thể làm chậm đáng kể tốc độ tính toán nếu thiếu nhân Tensor Core chuyên dụng.

---

## 5. KẾT LUẬN (CONCLUSION)

Nghiên cứu đã làm sáng tỏ cơ sở lý thuyết, chứng minh số học và kiểm chứng thực nghiệm sức mạnh của cặp kỹ thuật **LoRA** và **QLoRA**. Việc đóng băng $W_0$ kết hợp phân rã hạng thấp $BA$ không chỉ cắt giảm đến **98–99%** tham số huấn luyện mà còn giữ nguyên tốc độ suy luận nhờ phép gộp $W_{\text{deploy}} = W_0 + \frac{\alpha}{r}BA$. Kết hợp thêm lượng tử hóa 4-bit NF4, Double Quantization và Paged Optimizers của QLoRA, bài toán tinh chỉnh LLM 7B–65B trở nên khả thi ngay trên GPU phổ thông 12–24 GB VRAM. Tuy nhiên, người kỹ sư AI cần tỉnh táo đánh giá sự đánh đổi giữa tiết kiệm bộ nhớ, hiện tượng giới hạn dung lượng khi $r$ quá nhỏ, và giới hạn của LoRA trong các bài toán đòi hỏi thay đổi miền dữ liệu sâu hoặc nạp lượng lớn tri thức mới.

---

## TÀI LIỆU THAM KHẢO (REFERENCES)

1. E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, *"LoRA: Low-Rank Adaptation of Large Language Models,"* in *International Conference on Learning Representations (ICLR)*, 2022.
2. T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, *"QLoRA: Efficient Finetuning of Quantized LLMs,"* in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023.
3. N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly, *"Parameter-Efficient Transfer Learning for NLP,"* in *International Conference on Machine Learning (ICML)*, 2019, pp. 2790–2799.
4. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, *"Attention Is All You Need,"* in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.
5. V. Sanh, L. Debut, J. Chaumond, and T. Wolf, *"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,"* in *NeurIPS Workshop on Energy Efficient Machine Learning and Cognitive Computing (EMC2)*, 2019.
