"""
Semantic-Guided Temporal Interaction & Narrative Context Modules for DDVC.
Module hỗ trợ tương tác 2 chiều giữa Semantic Caption Queries (Q_cap) và Localization Queries (Q_loc),
thực hiện Visual Re-Attendance trên video features và mô hình hóa ngữ cảnh chuỗi sự kiện (Inter-Event Narrative Context).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ddvc.ops.modules import MSDeformAttn


class SemanticGuidedTemporalInteraction(nn.Module):
    """
    Module tương tác có hướng dẫn ngữ nghĩa (Cách B):
    1. Tiêm ngữ nghĩa hành động từ Q_cap vào Q_loc qua cổng thích ứng (Adaptive Gated Fusion).
    2. Tái tương tác thị giác (Visual Re-Attendance): Dùng Q_loc đã có ngữ nghĩa thực hiện
       Deformable Cross-Attention trên video features để thu thập bằng chứng thị giác ranh giới thời gian.
    """
    def __init__(self, d_model=512, n_levels=4, n_heads=8, n_points=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Cổng hòa trộn thích ứng (Gated Fusion)
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.proj_cap = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm_guide = nn.LayerNorm(d_model)

        # Visual Re-attendance qua Multi-Scale Deformable Attention (Cách B)
        self.visual_cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout = nn.Dropout(dropout)
        self.norm_visual = nn.LayerNorm(d_model)

        # FFN tinh chỉnh sau khi có visual evidence
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, q_loc, q_cap, reference_points, memory, temporal_shapes, level_start_index, padding_mask=None):
        """
        q_loc: [B, N_q, d_model] - Localization queries
        q_cap: [B, N_q, d_model] - Caption queries
        reference_points: [B, N_q, n_levels, 2 or 1]
        memory: [B, L_vid, d_model] - Video memory tokens
        """
        # 1. Semantic Injection via Gated Fusion (Ngữ nghĩa định hướng định vị)
        gate = self.gate_fc(torch.cat([q_loc, q_cap], dim=-1))
        cap_info = self.proj_cap(q_cap)
        q_loc_guided = self.norm_guide(q_loc + gate * cap_info)

        # 2. Visual Re-Attendance trên Video Memory (Cách B: Tìm bằng chứng khung hình video)
        visual_evidence = self.visual_cross_attn(
            q_loc_guided, reference_points, memory, temporal_shapes, level_start_index, padding_mask
        )
        q_loc_visual = self.norm_visual(q_loc_guided + self.dropout(visual_evidence))

        # 3. FFN Refinement
        q_loc_out = self.norm_out(q_loc_visual + self.dropout(self.ffn(q_loc_visual)))
        return q_loc_out


class InterEventNarrativeMemory(nn.Module):
    """
    Module mô hình hóa ngữ cảnh chuỗi sự kiện liên tiếp (Inter-Event Narrative Memory):
    - Nhận các caption queries Q_cap.
    - Sắp xếp các queries theo thứ tự xuất hiện thời gian (start timestamp).
    - Cho phép query sự kiện thứ k tiếp nhận ngữ cảnh của sự kiện thứ k-1 để tránh lặp từ và đảm bảo tính liền mạch đoạn văn.
    """
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_cap, pred_boxes):
        """
        q_cap: [B, N_q, d_model]
        pred_boxes: [B, N_q, 2] hoặc [B, N_q, 4] với [center, width]
        """
        if pred_boxes is None or q_cap.shape[1] <= 1:
            return q_cap

        bs, num_q, _ = q_cap.shape
        # Tính toán start time: center - 0.5 * width
        if pred_boxes.shape[-1] >= 2:
            start_times = pred_boxes[..., 0] - 0.5 * pred_boxes[..., 1]
        else:
            start_times = pred_boxes[..., 0]

        outputs = []
        for b in range(bs):
            # Sắp xếp query theo thứ tự thời gian tăng dần
            sort_idx = torch.argsort(start_times[b])
            inv_idx = torch.argsort(sort_idx)

            sorted_q = q_cap[b, sort_idx]  # [N_q, d_model]

            # Causal mask để sự kiện sau học từ sự kiện trước
            seq_len = num_q
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q_cap.device), diagonal=1).bool()

            # MultiheadAttention mong muốn [L, B, D]
            sorted_q_in = sorted_q.unsqueeze(1)  # [N_q, 1, d_model]
            attn_out, _ = self.self_attn(sorted_q_in, sorted_q_in, sorted_q_in, attn_mask=causal_mask)
            sorted_refined = self.norm(sorted_q_in + self.dropout(attn_out))
            sorted_refined = self.norm_out(sorted_refined + self.dropout(self.ffn(sorted_refined))).squeeze(1)

            # Khôi phục lại thứ tự query ban đầu
            original_order_q = sorted_refined[inv_idx].unsqueeze(0)
            outputs.append(original_order_q)

        return torch.cat(outputs, dim=0)
