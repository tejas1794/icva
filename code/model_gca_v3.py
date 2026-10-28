import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentCrossViewCoupler(nn.Module):
    def __init__(self, hidden_size, num_iterations=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations

        # Spatial pooling with a learned query anchor (Section 3.2.1)
        self.spatial_pooler = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Relation branches (Section 3.2.2). The parameter names are kept for
        # checkpoint compatibility; neither branch receives pose supervision.
        self.induce_trans = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2)
        )
        self.induce_rot = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2)
        )

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(hidden_size)

        # Zero-initialized residual scale (Section 3.3): at initialization the
        # adapted model is exactly the pretrained model.
        self.output_scale = nn.Parameter(torch.zeros(1))

        # Pooled frame representations, collected for the auxiliary loss
        self.collected_pools = []

    def reset_pools(self):
        self.collected_pools = []

    def forward(self, img_a, img_b):
        # img_a: [B, seq_len_a, H]
        # img_b: [B, seq_len_b, H]
        b, seq_a, h = img_a.shape
        _, seq_b, _ = img_b.shape

        # Spatial pooling
        q_anchor = self.pool_query.expand(b, -1, -1)
        pool_a, _ = self.spatial_pooler(q_anchor, img_a, img_a)
        pool_b, _ = self.spatial_pooler(q_anchor, img_b, img_b)

        pool_a = pool_a.squeeze(1)
        pool_b = pool_b.squeeze(1)

        # Kept on the graph for the auxiliary loss
        self.collected_pools.append((pool_a, pool_b))

        # Relation vector m
        pooled_cat = torch.cat([pool_a, pool_b], dim=-1)
        m_1 = self.induce_trans(pooled_cat)
        m_2 = self.induce_rot(pooled_cat)
        relation = torch.cat([m_1, m_2], dim=-1).unsqueeze(1)  # [B, 1, H]

        # K weight-shared refinement steps (Section 3.2.3)
        original_img_b = img_b.clone()
        refined_b = img_b

        for _ in range(self.num_iterations):
            query = refined_b + relation
            attn_out, _ = self.cross_attn(query, img_a, img_a)
            g = self.gate(torch.cat([refined_b, attn_out], dim=-1))
            refined_b = self.norm(refined_b + g * attn_out)

        # Identity-initialized residual (Equation 7)
        final_b = original_img_b + self.output_scale * refined_b

        return final_b
