# database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Sử dụng SQLite
DATABASE_URL = "sqlite:///./news1.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class NewsRecord(Base):
    __tablename__ = "news_history"

    id = Column(Integer, primary_key=True, index=True)
    
    # 1. Thông tin đầu vào
    content = Column(Text)                    # Tương đương cột 'claim' trong CSV
    meta_info = Column(String)                # JSON: likes, shares, comments
    
    # 2. Kết quả dự đoán chính
    prediction_label = Column(String)         # 'label'
    confidence_score = Column(Float)          # 'label_confidence'
    
    # 3. Thông tin bằng chứng (Best Evidence - Giống file CSV)
    evidence_title = Column(String, nullable=True)
    evidence_url = Column(String, nullable=True)
    evidence_source = Column(String, nullable=True)
    evidence_excerpt = Column(Text, nullable=True)
    evidence_retrieval_score = Column(Float, default=0.0)
    evidence_nli_label = Column(String, nullable=True)
    evidence_nli_confidence = Column(Float, default=0.0)
    
    # 4. Thông tin hệ thống
    layer_used = Column(String)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.now) # 'timestamp'

def init_db():
    # ⚠️ LỆNH NÀY SẼ XÓA SẠCH DỮ LIỆU CŨ ĐỂ TẠO BẢNG MỚI ĐÚNG CẤU TRÚC
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("✅ Đã làm mới Database (Drop & Create All)")