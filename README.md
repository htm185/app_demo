# PHÁT HIỆN THÔNG TIN SAI LỆCH TRÊN MẠNG XÃ HỘI TIẾNG VIỆT BẰNG AI

Dự án xây dựng hệ thống **phát hiện/kiểm chứng thông tin sai lệch** trên mạng xã hội tiếng Việt bằng AI theo hướng **đa tầng (adaptive)**, phân loại nội dung thành:

- **REAL**: thông tin đáng tin cậy
- **FAKE**: thông tin sai lệch
- **NEI / UNCERTAIN**: chưa đủ bằng chứng (không đủ thông tin để kết luận)

Hệ thống gồm:

- **Backend API**: FastAPI (`main_api.py`)
- **Frontend UI**: Streamlit (`frontend_ui.py`)
- **Lưu lịch sử**: SQLite + SQLAlchemy (`database.py`, DB file: `news1.db`)

---

## 1) Kiến trúc & luồng xử lý

### 1.1 Luồng tổng quan

1. Người dùng nhập **nội dung cần kiểm tra** và các chỉ số tương tác (**likes/comments/shares**) trên UI Streamlit.
2. UI gọi API `POST /verify`.
3. Backend chạy `adaptive_predict()` trong `backend_logic.py`:
   - **Layer 1 (lọc nhanh)**: dựa trên **văn phong + metadata tương tác**. Nếu độ tin cậy cao (mặc định `>= 0.90`) thì trả kết quả ngay.
   - **Layer 2 (đối chiếu bằng chứng)**: nếu Layer 1 chưa đủ tự tin → tìm bằng chứng từ web (Google CSE), cào nội dung trang, chạy NLI để đối chiếu “claim vs evidence”, sau đó tổng hợp kết quả.
4. Backend **lưu kết quả** vào bảng `news_history` trong SQLite.

### 1.2 Các thành phần chính

- `main_api.py`: định nghĩa API `/verify`, nhận input, gọi AI, lưu DB.
- `backend_logic.py`: pipeline chính (Layer 1/Layer 2), tổng hợp evidence, có **safety rules** để giảm false-positive.
- `ai_models.py`: định nghĩa kiến trúc model + hàm `load_models()`.
- `search_module.py`: tìm evidence qua Google Custom Search và chấm điểm kết quả.
- `evidence_processor.py`: cào nội dung trang (Trafilatura), chọn đoạn liên quan, chạy NLI.
- `save_verdicts.py`: (tuỳ chọn) lưu thêm kết quả ra `data/verdicts.jsonl` và `data/verdicts.csv`.
- `database.py`: SQLAlchemy model `NewsRecord`.

---

## 2) Yêu cầu hệ thống

- Python **3.9+** (khuyến nghị)
- (Tuỳ chọn) GPU/CUDA để tăng tốc PyTorch

Cài dependencies:

```bash
pip install -r requirements.txt
```

### Lưu ý về dependencies còn thiếu
Trong code hiện có sử dụng thêm một số thư viện nhưng chưa được khai báo trong `requirements.txt`, ví dụ:

- `trafilatura` (cào nội dung trang) – dùng trong `evidence_processor.py`
- `duckduckgo_search`, `pyvi` – dùng trong `keyword_extractor.py`

Nếu chạy gặp lỗi import, bạn có thể cài thêm:

```bash
pip install trafilatura duckduckgo_search pyvi
```

---

## 3) Cấu hình (Quan trọng)

### 3.1 Google Custom Search API (phục vụ Layer 2)

File: `search_module.py`

Bạn cần cấu hình 2 biến:

- `GOOGLE_API_KEY`
- `GOOGLE_CSE_ID`

Nếu chưa cấu hình, Layer 2 có thể không tìm được bằng chứng (hoặc trả về rỗng).

### 3.2 Model weights / scaler

File: `ai_models.py` và `backend_logic.py`

Hệ thống đang tìm các file (đường dẫn tương đối):

- `model/best_model.pt` (Model 1: Filter)
- `model/best_model.pth` (Model 2: Expert NLI)
- `model/meta_scaler.pkl` (chuẩn hoá metadata likes/comments/shares)

Nếu không có các file này, code vẫn chạy nhưng sẽ dùng **weights ngẫu nhiên** → kết quả không đáng tin.

---

## 4) Chạy Backend (FastAPI)

```bash
uvicorn main_api:app --reload --host 127.0.0.1 --port 8000
```

### API

- `POST /verify`

Ví dụ gọi API:

```bash
curl -X POST http://127.0.0.1:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "content": "...",
    "likes": 1200,
    "comments": 350,
    "shares": 800
  }'
```

---

## 5) Chạy Frontend (Streamlit)

Mở terminal khác:

```bash
streamlit run frontend_ui.py
```

UI mặc định gọi backend tại:

- `http://127.0.0.1:8000/verify`

Nếu bạn đổi host/port backend, hãy sửa biến `API_URL` trong `frontend_ui.py`.

---

## 6) Database

- SQLite file: `news1.db`
- Table: `news_history`
- Model: `NewsRecord` trong `database.py`

### Reset DB (cẩn thận)

Trong `database.py` có hàm:

```python
from database import init_db
init_db()
```

> Cảnh báo: `init_db()` sẽ **drop toàn bộ bảng** và tạo lại → **mất dữ liệu cũ**.

---

## 7) Troubleshooting

- **Không có evidence / Layer 2 không hoạt động**:
  - Kiểm tra `GOOGLE_API_KEY`/`GOOGLE_CSE_ID` trong `search_module.py`
  - Kiểm tra quota Google API (trường hợp HTTP 429)
- **Lỗi import thư viện**: cài thêm `trafilatura`, `pyvi`, `duckduckgo_search`.
- **Kết quả dự đoán không ổn định**: kiểm tra bạn đã đặt đúng các file weights trong thư mục `model/`.

---

## 8) Gợi ý cải thiện

- Dùng file `.env` (ví dụ `python-dotenv`) để quản lý API keys.
- Bổ sung đầy đủ thư viện vào `requirements.txt`.
- Tách cấu hình `CONFIDENCE_THRESHOLD`, `TEMPERATURE` sang file config.
- Thêm endpoint xem lịch sử (`GET /history`).

---

## License

Chưa khai báo.
