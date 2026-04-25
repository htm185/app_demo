# PHÁT HIỆN THÔNG TIN SAI LỆCH TRÊN MẠNG XÃ HỘI TIẾNG VIỆT BẰNG AI

Dự án xây dựng hệ thống **phát hiện/kiểm chứng thông tin sai lệch** trên mạng xã hội tiếng Việt bằng AI theo hướng **đa tầng (adaptive)**, phân loại nội d[...]

- **REAL**: thông tin đáng tin cậy
- **FAKE**: thông tin sai lệch
- **NEI / UNCERTAIN**: chưa đủ bằng chứng (không đủ thông tin để kết luận)



---

## 1) Kiến trúc & luồng xử lý

### 1.1 Luồng tổng quan

1. Người dùng nhập **nội dung cần kiểm tra** và các chỉ số tương tác (**likes/comments/shares**) trên UI Streamlit.
2. UI gọi API `POST /verify`.
3. Backend chạy `adaptive_predict()` trong `backend_logic.py`:
   - **Layer 1 (lọc nhanh)**: dựa trên **văn phong + metadata tương tác**. Nếu độ tin cậy cao (mặc định `>= 0.90`) thì trả kết quả ngay.
   - **Layer 2 (đối chiếu bằng chứng)**: nếu Layer 1 chưa đủ tự tin → tìm bằng chứng từ web (Google CSE), cào nội dung trang, chạy NLI để đối chiếu “claim [...]
4. Backend **lưu kết quả** vào bảng `news_history` trong SQLite.

### 1.2 Các thành phần chính

- `main_api.py`: định nghĩa API `/verify`, nhận input, gọi AI, lưu DB.
- `backend_logic.py`: pipeline chính (Layer 1/Layer 2), tổng hợp evidence, có **safety rules** để giảm false-positive.
- `ai_models.py`: định nghĩa kiến trúc model + hàm `load_models()`.
- `search_module.py`: tìm evidence qua Google Custom Search và chấm điểm kết quả.
- `evidence_processor.py`: cào nội dung trang (Trafilatura), chọn đoạn liên quan, chạy NLI.
- `database.py`: SQLAlchemy model `NewsRecord`.

---

## 2) Kết quả

### 2.1 So sánh hiệu năng (Layer 2)

<figure>
  <img src="assets/layer2-comparison.png" alt="Bảng so sánh hiệu năng trên tập kiểm thử ViFactCheck (Layer 2)" width="900" />
  <figcaption><i>Bảng 4-2: So sánh hiệu năng trên tập kiểm thử ViFactCheck (Layer 2).</i></figcaption>
</figure>

---

## 2) Yêu cầu hệ thống

- Python **3.9+** (khuyến nghị)
- (Tuỳ chọn) GPU/CUDA để tăng tốc PyTorch

Cài dependencies:

```bash
pip install -r requirements.txt
```

## 3) Chạy Backend (FastAPI)

```bash
uvicorn main_api:app --reload --host 127.0.0.1 --port 8000
```

---

## 5) Chạy Frontend (Streamlit)

Mở terminal khác:

```bash
streamlit run frontend_ui.py
```
