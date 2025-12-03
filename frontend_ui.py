import streamlit as st
import requests
import time

# ==============================================================================
# 1. CẤU HÌNH & CSS
# ==============================================================================
st.set_page_config(
    page_title="V-Check", 
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000/verify"


st.markdown("""
<style>
    /* Main container styling */
    .main { background-color: #f4f7f6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Card styling general */
    div.stStack > div[data-testid="stVerticalBlock"] > div.stHorizontalBlock {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    /* Headers */
    h1 { color: #2c3e50; font-weight: 700; }
    h3 { color: #34495e; }
    
    /* Result Badge Styled */
    .result-card {
        padding: 25px 20px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 20px;
        color: white;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Colors */
    .fake-bg { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); }
    .real-bg { background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%); border: 2px solid rgba(255, 255, 255, 0.3); }
    .nei-bg { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); }
    
    /* Text formatting inside cards */
    .result-icon { font-size: 50px; margin-bottom: 10px; display: block; }
    .result-title { font-size: 32px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; line-height: 1.2; }
    .result-subtitle { font-size: 16px; font-weight: 500; opacity: 0.95; margin-top: 5px; }

    /* Text Area tweak */
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        font-size: 16px;
        padding: 15px;
    }
    .stButton button { border-radius: 12px; height: 50px; font-size: 18px; font-weight: bold;}
    
    /* Tinh chỉnh tab */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Khởi tạo session state
if 'likes' not in st.session_state: st.session_state.likes = 0
if 'comments' not in st.session_state: st.session_state.comments = 0
if 'shares' not in st.session_state: st.session_state.shares = 0

# ==============================================================================
# 2. SIDEBAR
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/11698/11698479.png", width=80)
    st.title("⚙️ Tham số ngữ cảnh")
    st.caption("Các chỉ số tương tác giúp AI đánh giá mức độ lan truyền.")
    st.divider()
    
    st.session_state.likes = st.number_input("👍 Likes", min_value=0, value=st.session_state.likes, step=100)
    st.session_state.comments = st.number_input("💬 Comments", min_value=0, value=st.session_state.comments, step=50)
    st.session_state.shares = st.number_input("🔄 Shares", min_value=0, value=st.session_state.shares, step=50)
    
    st.divider()
    st.info("💡 **Lưu ý:** Tin giả thường có lượng Share/Comment cao bất thường.")

# ==============================================================================
# 3. GIAO DIỆN CHÍNH
# ==============================================================================

st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>🛡️ CỔNG KIỂM CHỨNG TIN TỨC AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #636e72; font-size: 18px; margin-bottom: 40px;'>Hệ thống phân tích tin tức đa tầng</p>", unsafe_allow_html=True)

col_input, col_space, col_output = st.columns([1, 0.05, 1]) 

# --- INPUT SECTION ---
with col_input:
    st.subheader("📝 Nội dung cần kiểm tra")
    with st.container(border=True):
        news_content = st.text_area(
            label="Input News",
            label_visibility="collapsed",
            height=250, 
            placeholder="Dán nội dung bài viết, tiêu đề hoặc một đoạn văn bản nghi ngờ vào đây...",
        )
        
        c1, c2, c3 = st.columns([0.5, 2, 0.5])
        with c2:
            submit_btn = st.button("🔍 BẮT ĐẦU PHÂN TÍCH", type="primary", use_container_width=True)

# ==============================================================================
# 4. XỬ LÝ VÀ HIỂN THỊ KẾT QUẢ
# ==============================================================================
with col_output:
    st.subheader("📊 Kết quả phân tích")
    
    if not submit_btn:
        st.info("👈 Vui lòng nhập nội dung và nhấn nút phân tích.")
        st.markdown(
            '<div style="text-align: center;"><img src="https://cdn-icons-png.flaticon.com/512/10354/10354693.png" width="200" style="opacity: 0.5;"></div>', 
            unsafe_allow_html=True
        )

    if submit_btn:
        if not news_content.strip():
            st.toast("⚠️ Vui lòng nhập nội dung!", icon="✍️")
        else:
            payload = {
                "content": news_content,
                "likes": st.session_state.likes,
                "comments": st.session_state.comments,
                "shares": st.session_state.shares
            }
            
            with st.spinner("🤖 Hệ thống đang xử lý phân tầng..."):
                try:
                    api_start = time.time()
                    response = requests.post(API_URL, json=payload)
                    api_end = time.time()
                    
                    if response.status_code == 200:
                        data = response.json()
                        res_data = data["result"]
                        
                        label = res_data["label"]
                        conf = res_data["confidence"]
                        layer = res_data["layer"]
                        reason = res_data["reason"]
                        
                        # --- TABS UI ---
                        tab1, tab2, tab3 = st.tabs(["TỔNG QUAN", "BẰNG CHỨNG", "LOG KỸ THUẬT"])
                        
                        # TAB 1: KẾT QUẢ CHÍNH
                        with tab1:
                            st.write("")
                            if label == "FAKE":
                                st.markdown(f"""
                                <div class="result-card fake-bg">
                                    <span class="result-icon">🚫</span>
                                    <div class="result-title">TIN GIẢ</div>
                                    <div class="result-subtitle">Nội dung có dấu hiệu sai lệch cao</div>
                                </div>""", unsafe_allow_html=True)
                                bar_color = "red"
                            elif label == "REAL":
                                st.markdown(f"""
                                <div class="result-card real-bg">
                                    <span class="result-icon">🛡️✅</span>
                                    <div class="result-title">TIN ĐÁNG TIN CẬY</div>
                                    <div class="result-subtitle">Nội dung được xác thực là chính xác</div>
                                </div>""", unsafe_allow_html=True)
                                bar_color = "green"
                            else:
                                st.markdown(f"""
                                <div class="result-card nei-bg">
                                    <span class="result-icon">🤔</span>
                                    <div class="result-title">CHƯA ĐỦ THÔNG TIN</div>
                                    <div class="result-subtitle">Cần thêm dữ liệu để kết luận</div>
                                </div>""", unsafe_allow_html=True)
                                bar_color = "orange"

                            st.write("#### Độ tin cậy của mô hình:")
                            st.progress(conf)
                            st.caption(f"Confidence Score: **{conf:.1%}**")
                            with st.container(border=True):
                                st.markdown(f"**💡 Lý do:** {reason}")

                        # TAB 2: BẰNG CHỨNG
                        with tab2:
                            st.write("")
                            if "evidence" in res_data and res_data["evidence"] and isinstance(res_data["evidence"], list):
                                st.success(f"🔎 Đã tìm thấy **{len(res_data['evidence'])}** nguồn tin.")
                                for i, ev in enumerate(res_data["evidence"]):
                                    with st.expander(f"📰 {ev.get('title', 'Nguồn tin')} ({ev.get('retrieval_score', 0):.2f})"):
                                        nli_lbl = ev.get('nli_label', 'NEI')
                                        if nli_lbl == "FAKE":
                                            st.error(f"❌ Nguồn này PHỦ NHẬN tin đầu vào.")
                                        elif nli_lbl == "REAL":
                                            st.success(f"✅ Nguồn này XÁC NHẬN tin đầu vào.")
                                        else:
                                            st.warning(f"⚠️ Nguồn này trung lập.")
                                        
                                        st.markdown(f"**URL:** [{ev.get('source')}]({ev.get('url')})")
                                        st.caption(f"...{ev.get('excerpt', '')}...")
                            else:
                                st.info("ℹ️ Không cần tra cứu bên ngoài (Layer 1 đã đủ độ tin cậy).")

                        # TAB 3: TRỰC QUAN HÓA
                        with tab3:
                            st.write("")
                            st.markdown(f"**Layer:** `{layer}` | **Time:** `{api_end - api_start:.3f}s`")
                            st.divider()
                            st.write("##### 🧠 Luồng quyết định:")
                            
                            # SỬA ĐỔI: Dùng cột để căn giữa và giới hạn chiều rộng biểu đồ
                            c1, c2, c3 = st.columns([1, 4, 1]) # Cột giữa chiếm 2/3 không gian
                            
                            with c2:
                                if "Layer 1" in layer:
                                    # Chuyển rankdir=LR (Ngang) và giảm fontsize
                                    st.graphviz_chart('''
                                    digraph {
                                        rankdir=LR; 
                                        bgcolor="transparent";
                                        node [shape=box, style="filled,rounded", fillcolor="white", color="#dfe6e9", fontname="Segoe UI", fontsize=11, height=0.4];
                                        edge [fontname="Segoe UI", fontsize=10, arrowsize=0.8];
                                        
                                        Input [label="Đầu vào", fillcolor="#ffeaa7", color="#fdcb6e"];
                                        Model1 [label="Layer 1\\n(Bộ lọc)", fillcolor="#74b9ff", color="#0984e3"];
                                        Result [label="Kết quả", style="filled", fillcolor="#55efc4", color="#00b894", fontcolor="white"];
                                        
                                        Input -> Model1 [penwidth=1.5];
                                        Model1 -> Result [label="Tin cậy ≥ 90%", color="#00b894", penwidth=2];
                                        Model1 -> Model2 [style="dotted", label="< 90%", color="#b2bec3"];
                                        Model2 [label="Layer 2", style="dashed", fontcolor="grey", color="#dfe6e9"];
                                    }
                                    ''', use_container_width=True)
                                else:
                                    # Chuyển rankdir=LR và tinh chỉnh layout gọn hơn
                                    st.graphviz_chart('''
                                    digraph {
                                        rankdir=LR;
                                        bgcolor="transparent";
                                        node [shape=box, style="filled,rounded", fillcolor="white", color="#dfe6e9", fontname="Segoe UI", fontsize=11, height=0.4];
                                        edge [fontname="Segoe UI", fontsize=10, arrowsize=0.8];

                                        Input [label="Đầu vào", fillcolor="#ffeaa7", color="#fdcb6e"];
                                        Model1 [label="Layer 1", fillcolor="#fab1a0", color="#e17055"];
                                        Search [label="Tìm kiếm", shape=ellipse, fillcolor="#a29bfe", color="#6c5ce7"];
                                        Model2 [label="Layer 2\\n(NLI)", fillcolor="#74b9ff", color="#0984e3"];
                                        Agg [label="Tổng hợp", shape=diamond, height=0.6, fontsize=10];
                                        Result [label="Kết quả", style="filled", fillcolor="#55efc4", color="#00b894", fontcolor="white"];
                                        
                                        Input -> Model1 [penwidth=1.5];
                                        Model1 -> Search [label="Thiếu tin cậy", color="#d63031"];
                                        Search -> Model2 [label="Top Evidence"];
                                        Model2 -> Agg [label="So khớp"];
                                        Agg -> Result [label="Quyết định", color="#00b894", penwidth=2];
                                    }
                                    ''', use_container_width=True)

                    else:
                        st.error(f"❌ Lỗi kết nối Server (Status: {response.status_code})")
                        
                except Exception as e:
                    st.error("🔌 Không thể kết nối Backend.")
                    st.code(e)