# search_module.py
import requests
import re
from typing import List, Dict


GOOGLE_API_KEY = ""
GOOGLE_CSE_ID = ""

# Danh sách nguồn uy tín để cộng điểm tin cậy
TRUSTED_SITES = [
    "vnexpress.net", "tuoitre.vn", "thanhnien.vn", "chinhphu.vn",
    "moh.gov.vn", "vtv.vn", "laodong.vn", "vietnamnet.vn",
    "dantri.com.vn", "suckhoedoisong.vn", "vinmec.com", "medlatec.vn",
    "plo.vn", "tienphong.vn", "nld.com.vn", "nhathuoclongchau.com.vn"
]

# Fallback import
try:
    from keyword_extractor import smart_keyword_extraction
except ImportError:
    def smart_keyword_extraction(text: str) -> str:
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        tokens = re.findall(r'\w+', text.lower())
        tokens = sorted([t for t in tokens if len(t) > 2], key=len, reverse=True)[:10]
        return " ".join(tokens)

def _extract_domain(url: str) -> str:
    try:
        parts = url.split('/')
        if len(parts) >= 3:
            return parts[2].lower()
    except:
        pass
    return "unknown"

def _tokens(text: str):
    return re.findall(r'\w+', (text or "").lower())

def _overlap_ratio(kwset, tokens):
    if not kwset: return 0.0
    common = kwset & set(tokens)
    return len(common) / len(kwset)

def _score_result(keywords: List[str], title: str, body: str, source_domain: str, full_query: str) -> float:
    kwset = set(k.lower() for k in keywords if k.strip())
    title_tokens = _tokens(title)
    body_tokens = _tokens(body)
    
    title_overlap = _overlap_ratio(kwset, title_tokens)
    body_overlap = _overlap_ratio(kwset, body_tokens)
    
    score = 0.7 * title_overlap + 0.3 * body_overlap
    
    qnorm = re.sub(r'[^\w\s]', '', full_query).strip().lower()
    if qnorm and len(qnorm) > 10 and qnorm in (title + " " + body).lower():
        score += 0.25
        
    if any(s in source_domain for s in TRUSTED_SITES):
        score += 0.20
        
    if len((body or "").strip()) < 50:
        score -= 0.30
        
    return max(0.0, min(1.0, score))

def _run_google_search(query: str, max_results: int = 6):
    if "DIEN_API_KEY" in GOOGLE_API_KEY:
        print("❌ LỖI: Chưa cấu hình GOOGLE_API_KEY trong search_module.py")
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_CSE_ID,
        'q': query,
        'num': min(max_results, 10),
        'lr': 'lang_vi'
    }
    
    results = []
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 429:
            print("⚠️ Google API: Hết hạn ngạch (Quota Exceeded).")
            return []
            
        data = resp.json()
        if 'items' in data:
            for item in data['items']:
                results.append({
                    'href': item.get('link'),
                    'title': item.get('title'),
                    'body': item.get('snippet', '')
                })
        elif 'error' in data:
            print(f"⚠️ Google API Error: {data['error'].get('message')}")
    except Exception as e:
        print(f"⚠️ Lỗi kết nối Google Search: {e}")
        
    return results

def search_google_evidence(query: str, max_results: int = 6, min_score: float = 0.25) -> List[Dict]:
    print(f"🔎 SEARCHING: {query[:60]}...")
    search_q = smart_keyword_extraction(query)
    keywords = [w.lower() for w in search_q.split() if w.strip()]
    
    # Chiến lược Query: Thử tìm bài Fact-check trước
    queries_to_try = []
    queries_to_try.append(query)
    
    check_words = ["sự thật", "thực hư", "có đúng", "lừa đảo", "giả mạo", "đính chính"]
    if not any(w in query.lower() for w in check_words):
        queries_to_try.append(f"Sự thật {query}")

    raw_results = []
    for q in queries_to_try:
        res = _run_google_search(q, max_results=max_results)
        raw_results.extend(res)
        if len(raw_results) >= max_results + 2: 
            break

    candidates = []
    blocklist = ['facebook', 'shopee', 'lazada', 'tiktok', 'youtube', 'pinterest', 'instagram', 'tiki']
    
    for res in raw_results:
        url = res.get('href', '') or ''
        title = res.get('title', '') or ''
        body = res.get('body', '') or ''
        source = _extract_domain(url)
        
        if any(x in url.lower() for x in blocklist): continue
            
        score = _score_result(keywords, title, body, source, query)
        is_trusted = any(s in source for s in TRUSTED_SITES)
        threshold = 0.15 if is_trusted else min_score
        
        if score >= threshold:
            candidates.append({
                "title": title,
                "url": url,
                "snippet": body,
                "source": source,
                "score": score + (0.1 if is_trusted else 0)
            })

    candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
    seen = set()
    final = []
    for c in candidates:
        if c['url'] in seen: continue
        seen.add(c['url'])
        final.append(c)
        if len(final) >= max_results: break
            
    print(f"✅ Found {len(final)} candidates.")
    return final
