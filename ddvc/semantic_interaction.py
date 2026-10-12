"""Lightweight semantic interaction and causal event-narrative modules."""
import torch
import torch.nn as nn
from ddvc.ops.modules import MSDeformAttn


class SemanticGuidedTemporalInteraction(nn.Module):
    """Exchange information between localization/caption queries, then revisit video evidence."""

    def __init__(self, d_model=512, n_levels=4, n_heads=8, n_points=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.loc_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.Sigmoid()
        )
        self.cap_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.Sigmoid()
        )
        self.proj_cap = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.proj_loc = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.norm_loc_guide = nn.LayerNorm(d_model)
        self.norm_cap_guide = nn.LayerNorm(d_model)

        self.visual_cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout = nn.Dropout(dropout)
        self.norm_visual = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model)
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, q_loc, q_cap, reference_points, memory, temporal_shapes, level_start_index, padding_mask=None):
        loc_cap = torch.cat([q_loc, q_cap], dim=-1)
        cap_loc = torch.cat([q_cap, q_loc], dim=-1)
        q_loc = self.norm_loc_guide(q_loc + self.loc_gate(loc_cap) * self.proj_cap(q_cap))
        q_cap = self.norm_cap_guide(q_cap + self.cap_gate(cap_loc) * self.proj_loc(q_loc))

        visual_evidence = self.visual_cross_attn(
            q_loc, reference_points, memory, temporal_shapes, level_start_index, padding_mask
        )
        q_loc = self.norm_visual(q_loc + self.dropout(visual_evidence))
        q_loc = self.norm_out(q_loc + self.dropout(self.ffn(q_loc)))
        return q_loc, q_cap


class InterEventNarrativeMemory(nn.Module):
    """Causal self-attention over caption queries ordered by predicted event start."""

    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model)
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_cap, pred_boxes):
        if pred_boxes is None or q_cap.shape[1] <= 1:
            return q_cap
        if pred_boxes.shape[-1] >= 2:
            start_times = pred_boxes[..., 0] - 0.5 * pred_boxes[..., 1]
        else:
            start_times = pred_boxes[..., 0]

        batch_size, num_queries, _ = q_cap.shape
        order = torch.argsort(start_times.detach(), dim=1, stable=True)
        inverse_order = torch.argsort(order, dim=1)
        ordered_q = q_cap.gather(1, order.unsqueeze(-1).expand(-1, -1, self.d_model))
        causal_mask = torch.triu(
            torch.ones(num_queries, num_queries, dtype=torch.bool, device=q_cap.device), diagonal=1
        )
        attended, _ = self.self_attn(ordered_q, ordered_q, ordered_q, attn_mask=causal_mask)
        ordered_q = self.norm(ordered_q + self.dropout(attended))
        ordered_q = self.norm_out(ordered_q + self.dropout(self.ffn(ordered_q)))
        return ordered_q.gather(1, inverse_order.unsqueeze(-1).expand(-1, -1, self.d_model))
