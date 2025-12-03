from duckduckgo_search import DDGS
import re
import sys

# Thử import thư viện NLP, nếu chưa có thì báo lỗi hướng dẫn cài
try:
    from pyvi import ViTokenizer, ViPosTagger
except ImportError:
    print("❌ LỖI: Thiếu thư viện 'pyvi'. Vui lòng chạy: pip install pyvi")
    sys.exit()

# Danh sách nguồn uy tín (Giữ nguyên)
TRUSTED_SITES = [
    "vnexpress.net", "tuoitre.vn", "thanhnien.vn", "chinhphu.vn",
    "moh.gov.vn", "vtv.vn", "laodong.vn", "vietnamnet.vn",
    "dantri.com.vn", "suckhoedoisong.vn", "vinmec.com", "medlatec.vn"
]

# Một stoplist nhỏ cho các token chức năng hay gây nhiễu
COMMON_NOISE = set([
    "nhiều", "tốt", "cho", "có", "không", "là", "được", "bị",
    "và", "của", "các", "một", "những", "thì", "cũng", "ra"
])

def smart_keyword_extraction(text, debug=False):
    """
    Trích xuất từ khóa dựa trên POS nhưng với:
    - Loại bỏ các từ chức năng/noise (COMMON_NOISE)
    - Ưu tiên danh từ (N*, Np), giữ các cụm như tim_mạch
    - Fallback về text gốc (clean) nếu keywords quá ít
    debug=True để in words/tags phục vụ debug
    """
    # 1. CLEAN: giữ nguyên để ViTokenizer token đúng (loại bỏ số, ký tự đặc biệt)
    cleaned = re.sub(r'\d+', '', text)
    # Không xóa dấu gạch dưới vì ViTokenizer sẽ tạo underscore cho các cụm
    cleaned = re.sub(r'[^\w\s]', '', cleaned)

    # 2. TOKENIZE + POS
    tokenized_text = ViTokenizer.tokenize(cleaned)  # ví dụ: "tim_mạch"
    try:
        words, tags = ViPosTagger.postagging(tokenized_text)
    except Exception:
        # Nếu pos tagging fail, fallback: split tokenized_text
        words = tokenized_text.split()
        tags = [''] * len(words)

    if debug:
        print("DEBUG POS:", list(zip(words, tags)))

    core_keywords = []

    for word, tag in zip(words, tags):
        # Giữ nếu là danh từ/proper noun/số/từ chuyên môn
        # tag có thể là 'N', 'Np', 'Nc', 'V', 'M' ... nên check startswith
        keep = False
        if tag:
            if tag.startswith('N') or tag.startswith('Np') or tag.startswith('Nc') or tag.startswith('M'):
                keep = True
            # optionally keep verbs if not enough nouns later
        # Sanitize word
        word_clean = word.replace("_", " ").strip().lower()
        if word_clean in COMMON_NOISE:
            keep = False
        if keep and word_clean:
            core_keywords.append(word_clean)

    # Nếu quá ít keywords, mở rộng: thêm verbs (tag V) trừ các helper verbs
    if len(core_keywords) < 2:
        for word, tag in zip(words, tags):
            word_clean = word.replace("_", " ").strip().lower()
            if word_clean in COMMON_NOISE:
                continue
            if tag and tag.startswith('V') and word_clean not in core_keywords:
                core_keywords.append(word_clean)
            if len(core_keywords) >= 2:
                break

    # Final fallback: nếu vẫn ít (<2), trả về nguyên câu đã clean (không tokenized)
    if not core_keywords or len(core_keywords) < 2:
        # dùng phiên bản đã cleaned (loại số/ký tự đặc biệt)
        fallback = re.sub(r'\s+', ' ', cleaned).strip()
        if debug:
            print("DEBUG fallback to cleaned text:", fallback)
        return fallback

    # Trả về chuỗi từ khóa (ưu tiên giữ cụm từ như "tim mạch" đã nối)
    return " ".join(core_keywords)