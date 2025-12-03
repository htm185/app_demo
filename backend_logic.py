import torch
import torch.nn.functional as F
import joblib
import numpy as np
import os
import concurrent.futures
from ai_models import load_models, device
from search_module import search_google_evidence
from evidence_processor import fetch_page_text, run_nli_on_pair
from save_verdicts import save_verdict

# --- CẤU HÌNH ---
SCALER_PATH = "model/meta_scaler.pkl"
CONFIDENCE_THRESHOLD = 0.90
TEMPERATURE = 2.0
AUTO_SAVE = True
JSONL_PATH = "data/verdicts.jsonl"
CSV_PATH = "data/verdicts.csv"

print(f"⏳ Đang khởi tạo hệ thống trên thiết bị: {device}...")
(m1, t1), (m2, t2) = load_models()

scaler = None
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
        print("✅ Đã tải meta_scaler.pkl")
    except Exception: pass

# --- HÀM SAFETY RULE (MỚI) ---
def apply_safety_rules(claim_text, result):
    """
    Luật hậu xử lý: Nếu Claim khẳng định 100% (tuyệt đối)
    mà bằng chứng chỉ là 'hỗ trợ', thì lật thành FAKE.
    """
    claim_lower = claim_text.lower()
    sensational_keywords = ["100%", "tuyệt đối", "bách bệnh", "thần dược", "dứt điểm", "khỏi hẳn", "cam kết", "hoàn toàn"]
    safe_keywords = ["hỗ trợ", "cải thiện", "giảm nguy cơ", "phòng ngừa", "ngăn chặn", "có thể", "lời khuyên"]
    
    has_sensational = any(kw in claim_lower for kw in sensational_keywords)
    
    if has_sensational and result["label"] == "REAL":
        evidence_snippets = " ".join([e.get("excerpt", "").lower() for e in result["evidence"]])
        if any(safe_word in evidence_snippets for safe_word in safe_keywords):
            print("⚠️ SAFETY RULE TRIGGERED: Phát hiện phóng đại!")
            result["label"] = "FAKE"
            result["confidence"] = 0.96
            result["reason"] = "Cảnh báo: Nội dung chứa từ ngữ khẳng định tuyệt đối (100%, bách bệnh) nhưng bằng chứng khoa học chỉ dừng ở mức 'hỗ trợ' hoặc 'giảm nguy cơ'."
    return result

def aggregate_evidence_nli(evidence_nli_entries):
    total_weight = 0.0
    agg = np.zeros(3, dtype=float)
    for ev in evidence_nli_entries:
        w = float(ev.get("score", 0.0))
        probs = np.array(ev.get("probs", [0.0,0.0,0.0]), dtype=float)
        agg += w * probs
        total_weight += w
    
    if total_weight <= 0: agg_probs = np.ones(3) / 3
    else: agg_probs = agg / total_weight
    
    pred_idx = int(np.argmax(agg_probs))
    labels_map = {0: "REAL", 1: "FAKE", 2: "NEI"}
    return labels_map.get(pred_idx, "NEI"), float(agg_probs[pred_idx]), agg_probs.tolist()

def _maybe_save_result(claim: str, result: dict):
    if AUTO_SAVE:
        try: save_verdict(claim, result, jsonl_path=JSONL_PATH, csv_path=CSV_PATH)
        except Exception: pass

def adaptive_predict(text, likes, comments, shares, debug=False):
    # LAYER 1
    if len(text.split()) >= 15:
        inputs1 = t1(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs1 = {k: v.to(device) for k, v in inputs1.items()}
        
        meta_val = np.array([[likes, comments, shares]], dtype=np.float32)
        if scaler: meta_val = scaler.transform(meta_val)
        meta_tensor = torch.tensor(meta_val, dtype=torch.float).to(device)

        with torch.no_grad():
            logits1 = m1(inputs1['input_ids'], inputs1['attention_mask'], meta_tensor)
            probs1 = F.softmax(logits1 / TEMPERATURE, dim=1)
            conf1, pred1 = torch.max(probs1, dim=1)
            conf1 = conf1.item()
            lbl1 = "FAKE" if pred1.item() == 1 else "REAL"

        if conf1 >= CONFIDENCE_THRESHOLD:
            res = {"label": lbl1, "confidence": conf1, "layer": "Layer 1: Sàng lọc sơ cấp", "reason": "Dựa trên văn phong và chỉ số tương tác.", "evidence": []}
            _maybe_save_result(text, res)
            return res

    # LAYER 2
    evidence_list = search_google_evidence(text, max_results=6, min_score=0.25)
    if not evidence_list:
        res = {"label": "UNCERTAIN", "confidence": 0.0, "layer": "Layer 2: Tra cứu thất bại", "reason": "Không tìm thấy nguồn tin liên quan.", "evidence": []}
        _maybe_save_result(text, res)
        return res

    target_evidence = evidence_list[:4]
    fetched_data = []
    
    def process_fetch(ev):
        return ev, fetch_page_text(ev["url"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ev = {executor.submit(process_fetch, ev): ev for ev in target_evidence}
        for future in concurrent.futures.as_completed(future_to_ev):
            try: fetched_data.append(future.result())
            except: pass

    per_evidence_results = []
    for ev, page_text in fetched_data:
        nli_out = run_nli_on_pair(text, page_text, m2, t2, device)
        per_evidence_results.append({
            "title": ev.get("title"), "url": ev.get("url"), "source": ev.get("source"),
            "retrieval_score": float(ev.get("score", 0.0)),
            "nli_label": nli_out["pred_label"], "nli_confidence": nli_out["pred_confidence"],
            "nli_probs": nli_out["probs"], "excerpt": nli_out["excerpt"]
        })

    per_evidence_results.sort(key=lambda x: x["retrieval_score"], reverse=True)
    agg_label, agg_conf, agg_probs = aggregate_evidence_nli(per_evidence_results)

    final_result = {
        "label": agg_label, "confidence": agg_conf,
        "layer": "Layer 2: Retrieval + NLI (Verified)",
        "reason": "Tổng hợp đối chiếu từ nhiều nguồn tin.",
        "evidence": per_evidence_results,
        "aggregator_probs": agg_probs,
    }

    # ÁP DỤNG LUẬT SAFETY
    final_result = apply_safety_rules(text, final_result)

    _maybe_save_result(text, final_result)
    return final_result