# main_api.py
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
import time
import json
import database
from backend_logic import adaptive_predict

app = FastAPI(title="Adaptive News Verification API")

# Khởi tạo DB (Sẽ xóa cũ và tạo mới bảng do code trong database.py)
# database.init_db()

class NewsRequest(BaseModel):
    content: str
    likes: int
    comments: int
    shares: int

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

def extract_best_evidence(evidence_list):
    """Chọn bằng chứng tốt nhất để lưu vào cột SQL """
    if not evidence_list:
        return {}
    
    # Ưu tiên: Không phải NEI -> Điểm tìm kiếm cao -> Độ tin cậy cao
    # (Giả định evidence_list đã được sort theo retrieval_score từ backend)
    best_ev = evidence_list[0] 
    
    # Nếu muốn logic phức tạp hơn (như ưu tiên label != NEI), có thể sort lại ở đây
    # sorted_list = sorted(evidence_list, key=lambda x: (x.get('nli_label') == 'NEI', -x.get('retrieval_score', 0)))
    # best_ev = sorted_list[0]
    
    return best_ev

@app.post("/verify")
async def verify_news_endpoint(request: NewsRequest, db: Session = Depends(get_db)):
    start_time = time.time()
    
    # 1. Gọi AI xử lý
    result = adaptive_predict(
        request.content, 
        request.likes, 
        request.comments, 
        request.shares
    )
    
    process_time = time.time() - start_time
    
    # 2. Lấy thông tin bằng chứng tốt nhất (nếu có)
    evidence_list = result.get("evidence", [])
    best_ev = extract_best_evidence(evidence_list)
    
    # 3. Lưu vào SQL (Mapping đầy đủ các trường)
    db_record = database.NewsRecord(
        content=request.content,
        meta_info=json.dumps({"likes": request.likes, "comments": request.comments, "shares": request.shares}),
        
        prediction_label=result["label"],
        confidence_score=result["confidence"],
        
        # Mapping các cột bằng chứng mới
        evidence_title=best_ev.get("title", ""),
        evidence_url=best_ev.get("url", ""),
        evidence_source=best_ev.get("source", ""),
        evidence_excerpt=best_ev.get("excerpt", ""),
        evidence_retrieval_score=best_ev.get("retrieval_score", 0.0),
        evidence_nli_label=best_ev.get("nli_label", ""),
        evidence_nli_confidence=best_ev.get("nli_confidence", 0.0),
        
        layer_used=result["layer"],
        processing_time=process_time
    )
    
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    return {
        "id": db_record.id,
        "result": result,
        "processing_time": process_time
    }