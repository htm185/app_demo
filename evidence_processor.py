import requests
import re
import textwrap
import trafilatura
import torch
import torch.nn.functional as F

def fetch_page_text(url, timeout=5):
    """Cào dữ liệu bằng Trafilatura"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0"}
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            try:
                resp = requests.get(url, timeout=timeout, headers=headers)
                downloaded = resp.text
            except: return ""

        if not downloaded: return ""
        
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False, no_fallback=False)
        if not text: return ""
        
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    except Exception as e:
        print(f"⚠️ Error scraping {url}: {e}")
        return ""

def excerpt_text(text, max_chars=400):
    t = (text or "").strip()
    if not t: return ""
    return textwrap.shorten(t, width=max_chars, placeholder=" ...")

def find_best_segment(claim, full_text, window_size=350, step=150):
    """Sliding Window: Tìm đoạn văn liên quan nhất"""
    if not full_text: return ""
    words = full_text.split()
    if len(words) <= window_size: return full_text

    claim_tokens = set(re.findall(r'\w+', claim.lower()))
    bonus_keywords = ["sai", "không", "giả", "lừa", "đính chính", "bác bỏ", "sự thật", "nguy hiểm", "khẳng định", "lưu ý"]
    
    best_segment = ""
    max_score = -1
    
    for i in range(0, len(words), step):
        segment_words = words[i : i + window_size]
        segment_text = " ".join(segment_words)
        segment_lower = segment_text.lower()
        
        segment_tokens = set(re.findall(r'\w+', segment_lower))
        score = len(claim_tokens.intersection(segment_tokens)) * 2
        
        for kw in bonus_keywords:
            if kw in segment_lower: score += 1.5
        
        if score > max_score:
            max_score = score
            best_segment = segment_text
            
    return best_segment if best_segment else " ".join(words[:window_size])

def run_nli_on_pair(claim, evidence_text, m2, t2, device, max_stmt_tokens=100, max_ctx_tokens=350, debug=False):
    """Chạy NLI với Truncation an toàn"""
    best_context = find_best_segment(claim, evidence_text)
    
    stmt_tokens = t2(claim, add_special_tokens=False, truncation=True, max_length=max_stmt_tokens)["input_ids"]
    ctx_tokens = t2(best_context, add_special_tokens=False, truncation=True, max_length=max_ctx_tokens)["input_ids"]

    model_max_len = getattr(t2, "model_max_length", 512)
    if model_max_len > 10000: model_max_len = 512
    
    cls_id = getattr(t2, "cls_token_id", 0)
    sep_id = getattr(t2, "sep_token_id", 2)

    input_ids = [cls_id] + stmt_tokens + [sep_id] + ctx_tokens + [sep_id]

    if len(input_ids) > model_max_len:
        excess = len(input_ids) - model_max_len
        ctx_tokens = ctx_tokens[:-excess]
        input_ids = [cls_id] + stmt_tokens + [sep_id] + ctx_tokens + [sep_id]

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_tensor).to(device)

    stmt_len = len(stmt_tokens)
    ctx_len = len(ctx_tokens)
    s_start = torch.tensor([1], dtype=torch.long).to(device)
    s_end = torch.tensor([1 + stmt_len], dtype=torch.long).to(device)
    c_start = torch.tensor([1 + stmt_len + 1], dtype=torch.long).to(device)
    c_end = torch.tensor([1 + stmt_len + 1 + ctx_len], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = m2(input_tensor, attention_mask, s_start, s_end, c_start, c_end)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        pred_conf = float(probs[pred_idx])
        labels_map = {0: "REAL", 1: "FAKE", 2: "NEI"}
        pred_label = labels_map.get(pred_idx, "NEI")

    return {
        "probs": probs.tolist(),
        "pred_label": pred_label,
        "pred_confidence": pred_conf,
        "excerpt": excerpt_text(best_context, max_chars=300)
    }