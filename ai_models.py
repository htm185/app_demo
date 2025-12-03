import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- Dòng này quan trọng để sửa lỗi NameError 'F'
from transformers import AutoModel, AutoTokenizer
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
MODEL1_PATH = "model/best_model.pt"      # File Model 1 (PhoBERT + Meta)
MODEL2_PATH = "model/best_model.pth"  # File Model 2 (Expert NLI)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. MODEL 1: PHOBERT + METADATA (Cấu trúc chuẩn khớp với best_model.pt)
# ==============================================================================
class MultiModalFilterModel(nn.Module):
    def __init__(self, model_name="vinai/phobert-base", meta_dim=3, hidden_size=768, num_labels=2, dropout_rate=0.3):
        super(MultiModalFilterModel, self).__init__()
        
        # Backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size
        
        # Layer Weights (Học trọng số cho 4 lớp cuối)
        self.layer_weights = nn.Parameter(torch.ones(4) / 4)
        
        # LSTM
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size // 2,
                            num_layers=1, batch_first=True, bidirectional=True)
        
        # CNN (Conv1d)
        kernel_size = 3
        self.conv1d = nn.Conv1d(self.hidden_size, self.hidden_size,
                                kernel_size=kernel_size, padding=kernel_size//2)
        
        # Metadata Branch
        self.meta_net = nn.Sequential(
            nn.Linear(meta_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size + 16, num_labels)

    def _get_weighted_layers(self, outputs):
        # Lấy hidden states từ output của PhoBERT
        all_layers = outputs.hidden_states
        # Stack 4 lớp cuối lại: [Batch, Seq, Hidden, 4]
        last_4_layers = torch.stack(all_layers[-4:], dim=-1) 
        # Tính softmax trên trọng số học được
        weights = F.softmax(self.layer_weights, dim=0)
        # Nhân trọng số và cộng gộp
        return (last_4_layers * weights).sum(dim=-1)

    def forward(self, input_ids, attention_mask, meta_features):
        # 1. Text Branch
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, 
                                output_hidden_states=True, return_dict=True)
        
        # Tổng hợp 4 layer cuối
        features = self._get_weighted_layers(outputs)
        
        # Qua LSTM
        features, _ = self.lstm(features)
        
        # Qua CNN (Permute để đúng chiều cho Conv1d: [Batch, Channel, Seq])
        features_permuted = features.permute(0, 2, 1)
        cnn_out = self.conv1d(features_permuted)
        # Activation + Permute lại
        cnn_out = F.relu(cnn_out).permute(0, 2, 1)
        
        # Residual connection: LSTM + CNN
        text_features = features + cnn_out
        
        # Global Max Pooling (có che mặt nạ padding)
        mask_expanded = attention_mask.unsqueeze(-1).expand(text_features.size()).float()
        # Gán giá trị cực nhỏ vào vị trí padding để Max Pooling không lấy phải nó
        text_features = text_features.masked_fill(mask_expanded == 0, -1e9)
        pooled_text, _ = torch.max(text_features, dim=1) # [Batch, 768]
        
        # 2. Metadata Branch
        processed_meta = self.meta_net(meta_features) # [Batch, 16]
        
        # 3. Fusion & Classification
        combined_features = torch.cat((pooled_text, processed_meta), dim=1)
        combined_features = self.dropout(combined_features)
        logits = self.classifier(combined_features)
        
        return logits

# ==============================================================================
# 2. MODEL 2: EXPERT NLI (Wrapper để khớp checkpoint dictionary)
# ==============================================================================
class SingleBackboneEncoder(nn.Module):
    def __init__(self, backbone, hidden_size=1024, dropout_rate=0.2):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.ln_stmt = nn.LayerNorm(self.hidden_size)
        self.ln_ctx = nn.LayerNorm(self.hidden_size)

    def _create_segment_masks(self, seq_len, stmt_start, stmt_end, ctx_start, ctx_end, device):
        indices = torch.arange(seq_len, device=device).unsqueeze(0)
        stmt_mask = (indices >= stmt_start.unsqueeze(1)) & (indices < stmt_end.unsqueeze(1))
        ctx_mask = (indices >= ctx_start.unsqueeze(1)) & (indices < ctx_end.unsqueeze(1))
        return stmt_mask.float(), ctx_mask.float()

    def _masked_mean(self, H, mask):
        mask_expanded = mask.unsqueeze(-1)
        numerator = (H * mask_expanded).sum(dim=1)
        denominator = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return numerator / denominator

    def forward(self, input_ids, attention_mask, stmt_start, stmt_end, ctx_start, ctx_end):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        H = outputs.last_hidden_state
        cls_output = outputs.pooler_output
        
        batch_size, seq_len, _ = H.shape
        stmt_mask, ctx_mask = self._create_segment_masks(seq_len, stmt_start, stmt_end, ctx_start, ctx_end, H.device)
        
        pooled_stmt = self._masked_mean(H, stmt_mask)
        pooled_ctx = self._masked_mean(H, ctx_mask)
        
        pooled_stmt = self.dropout(self.ln_stmt(pooled_stmt))
        pooled_ctx = self.dropout(self.ln_ctx(pooled_ctx))
        
        return cls_output, pooled_stmt, pooled_ctx

class MultiTaskHead(nn.Module):
    def __init__(self, hidden_size=1024, num_labels=3, num_topics=53, dropout=0.3):
        super().__init__()
        self.shared = nn.Linear(hidden_size * 5, hidden_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_size) 
        self.dropout = nn.Dropout(dropout)
        self.cls_label = nn.Linear(hidden_size, num_labels)
        
    def forward(self, cls_output, pooled_stmt, pooled_ctx):
        abs_diff = torch.abs(pooled_stmt - pooled_ctx)
        element_prod = pooled_stmt * pooled_ctx
        h = torch.cat([cls_output, pooled_stmt, pooled_ctx, abs_diff, element_prod], dim=-1)
        
        h = self.shared(self.dropout(h))
        h = self.relu(h)
        h = self.bn(h) # Tạm tắt BN để tránh lỗi batch=1 khi inference
        
        logits_label = self.cls_label(self.dropout(h))
        return logits_label

class ExpertModelWrapper(nn.Module):
    """Class gộp cả Encoder và Head để dễ gọi"""
    def __init__(self, model_name="xlm-roberta-large"):
        super().__init__()
        backbone = AutoModel.from_pretrained(model_name)
        self.encoder = SingleBackboneEncoder(backbone)
        self.head = MultiTaskHead() # hidden_size mặc định 1024 cho large

    def forward(self, input_ids, attention_mask, stmt_start, stmt_end, ctx_start, ctx_end):
        cls, ps, pc = self.encoder(input_ids, attention_mask, stmt_start, stmt_end, ctx_start, ctx_end)
        logits = self.head(cls, ps, pc)
        return logits

# ==============================================================================
# 3. HÀM LOAD MODEL
# ==============================================================================
def load_models():
    print(f"⚙️ Thiết bị sử dụng: {device}")

    # --- MODEL 1 ---
    print("🔄 Đang tải Model 1 (Filter)...")
    t1 = AutoTokenizer.from_pretrained("vinai/phobert-base")
    m1 = MultiModalFilterModel()
    
    if os.path.exists(MODEL1_PATH):
        try:
            # Load strict=True
            m1.load_state_dict(torch.load(MODEL1_PATH, map_location=device), strict=True)
            print("✅ Model 1 OK (Strict Loaded)")
        except Exception as e:
            print(f"⚠️ Lỗi strict load Model 1: {e}")
            try:
                # Thử strict=False
                m1.load_state_dict(torch.load(MODEL1_PATH, map_location=device))
                print("✅ Model 1 OK (Relaxed Loaded)")
            except Exception as e2:
                print(f"❌ Không thể load Model 1: {e2}")
    else:
        print("ℹ️ Không tìm thấy file Model 1, dùng weight ngẫu nhiên.")
    
    m1.to(device).eval()

    # --- MODEL 2 ---
    print("🔄 Đang tải Model 2 (Expert)...")
    t2 = AutoTokenizer.from_pretrained("xlm-roberta-large") 
    m2 = ExpertModelWrapper(model_name="xlm-roberta-large")
    
    if os.path.exists(MODEL2_PATH):
        try:
            checkpoint = torch.load(MODEL2_PATH, map_location=device)
            # Load Encoder
            m2.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            # Load Head
            m2.head.load_state_dict(checkpoint['head_state_dict'], strict=False) 
            
            print(f"✅ Model 2 OK (Epoch {checkpoint.get('epoch', '?')}, F1: {checkpoint.get('val_f1', '?')})")
        except Exception as e:
             print(f"❌ Lỗi load Model 2: {e}")
    else:
        print("ℹ️ Không tìm thấy file Model 2, dùng weight ngẫu nhiên.")

    m2.to(device).eval()
    
    return (m1, t1), (m2, t2)