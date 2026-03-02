"""
RegMamba: Complete Model
================================================================================
完整流程:
    Stage 1: Input [B,N,3] -> PatchEmbedding -> tokens [B,L,D], centroids [B,L,3]
    Stage 2: tokens -> BiMambaBackbone -> features [B,L,D] + intermediate_feats
    Stage 3: features + centroids -> BATInteractionModule -> L_bat [B,L,L,2D+12]
    Stage 4: L_bat -> BATPoseDecoder -> quaternion [B,4], translation [B,3]
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional

# 导入各模块 (假设在同一目录下)
# from .patch_embedding import PatchEmbedding, RegMambaConfig
# from .mamba_backbone import BiMambaBackbone
# from .bat_module import BATInteractionModule
# from .pose_regression import BATPoseDecoder


# =============================================================================
# 配置类
# =============================================================================

class RegMambaConfig:
    """RegMamba 全局配置"""
    def __init__(
        self,
        n_points: int = 14400,      # N: 输入点数
        patch_size: int = 32,        # P: Patch大小
        stride: int = 16,            # S: 步长 (50%重叠率)
        d_model: int = 256,          # D/C: 特征维度
        n_mamba_layers: int = 4,     # Mamba层数
        n_heads: int = 8,            # 注意力头数
        d_state: int = 16,           # Mamba状态维度
        d_conv: int = 4,             # Mamba卷积核大小
        expand: int = 2,             # Mamba扩展因子
        dropout: float = 0.1,        # Dropout率
        morton_resolution: int = 1024,
    ):
        self.n_points = n_points
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.n_mamba_layers = n_mamba_layers
        self.n_heads = n_heads
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout = dropout
        self.morton_resolution = morton_resolution
        
        # 计算序列长度 L = (N - P) / S + 1
        self.seq_len = (n_points - patch_size) // stride + 1
        
        # BAT 特征维度: 2C + 12
        self.bat_feature_dim = 2 * d_model + 12
        
    def __repr__(self):
        return (f"RegMambaConfig(\n"
                f"  N={self.n_points}, P={self.patch_size}, S={self.stride},\n"
                f"  L={self.seq_len}, D={self.d_model},\n"
                f"  BAT_dim={self.bat_feature_dim},\n"
                f"  n_mamba_layers={self.n_mamba_layers}, n_heads={self.n_heads}\n"
                f")")


# =============================================================================
# 为了方便展示，这里将所有模块整合到一个文件中
# 实际项目中建议拆分成多个文件
# =============================================================================

# ---------- Stage 1: Patch Embedding (简化版，完整版见前面的代码) ----------

def z_order_sort(points: torch.Tensor, resolution: int = 1024):
    """Z-Order 排序"""
    def interleave_bits_3d(x, y, z):
        def expand_bits(v):
            v = v.long()
            v = (v | (v << 32)) & 0x1f00000000ffff
            v = (v | (v << 16)) & 0x1f0000ff0000ff
            v = (v | (v << 8)) & 0x100f00f00f00f00f
            v = (v | (v << 4)) & 0x10c30c30c30c30c3
            v = (v | (v << 2)) & 0x1249249249249249
            return v
        return expand_bits(x) * 4 + expand_bits(y) * 2 + expand_bits(z)
    
    B, N, _ = points.shape
    p_min = points.min(dim=1, keepdim=True)[0]
    p_max = points.max(dim=1, keepdim=True)[0]
    scale = (p_max - p_min).clamp(min=1e-8)
    normalized = (points - p_min) / scale
    quantized = (normalized * (resolution - 1)).long().clamp(0, resolution - 1)
    
    morton = interleave_bits_3d(quantized[:,:,0], quantized[:,:,1], quantized[:,:,2])
    sort_indices = morton.argsort(dim=1)
    expanded_indices = sort_indices.unsqueeze(-1).expand(-1, -1, 3)
    sorted_points = torch.gather(points, dim=1, index=expanded_indices)
    
    return sorted_points, sort_indices


class SlidingWindowUnfold(nn.Module):
    """滑动窗口展开点云为 Patch (50% 重叠),输出 [B,L,P,3]（L = 序列长度，P=Patch 大小）"""
    def __init__(self, patch_size: int = 32, stride: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        P, S = self.patch_size, self.stride
        L = (N - P) // S + 1
        
        x_t = x.transpose(1, 2)
        patches = x_t.unfold(dimension=2, size=P, step=S)
        patches = patches.permute(0, 2, 3, 1).contiguous()
        
        return patches


class LocalGeometryEncoder(nn.Module):
    """局部几何编码器"""
    # Patch 中心点归一化→MLP 提取特征→MaxPooling 聚合，输出 [B,L,128]
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, patches: torch.Tensor):
        B, L, P, C = patches.shape
        centroids = patches.mean(dim=2, keepdim=True)
        local_points = patches - centroids
        
        local_flat = local_points.reshape(B * L * P, 3)
        feat_flat = self.mlp(local_flat)
        feat = feat_flat.reshape(B, L, P, self.hidden_dim)
        geometric_feat = torch.max(feat, dim=2)[0]
        
        return geometric_feat, centroids.squeeze(2)


class PositionalEncoding(nn.Module):
    """位置编码"""
    # 对 Patch 中心点做 MLP 编码，输出 [B,L,128]
    def __init__(self, pe_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, pe_dim),
        )
        
    def forward(self, centroids: torch.Tensor):
        return self.mlp(centroids)


class PatchEmbedding(nn.Module):
    """Patch Embedding 完整模块"""
    # 整合上述模块
    #Z 排序→切分 Patch→几何 + 位置特征拼接→投影到 D 维，输出 tokens [B,L,D]、centroids [B,L,3]（Patch 中心点）
    def __init__(self, config: RegMambaConfig):
        super().__init__()
        self.config = config
        self.unfold = SlidingWindowUnfold(config.patch_size, config.stride)
        self.local_encoder = LocalGeometryEncoder(hidden_dim=128)
        self.positional_encoding = PositionalEncoding(pe_dim=128)
        self.projection = nn.Linear(256, config.d_model)
        
    def forward(self, points: torch.Tensor):
        sorted_points, sort_indices = z_order_sort(points, self.config.morton_resolution)
        patches = self.unfold(sorted_points)
        geometric_feat, centroids = self.local_encoder(patches)
        positional_feat = self.positional_encoding(centroids)
        combined_feat = torch.cat([geometric_feat, positional_feat], dim=-1)
        tokens = self.projection(combined_feat)
        
        return tokens, centroids, sort_indices


# ---------- Stage 2: Bi-Directional Mamba Backbone ----------

from mamba_ssm import Mamba

class BiMambaBlock(nn.Module):
    """双向 Mamba 块"""
    # 前向 Mamba + 反向 Mamba（翻转序列）→特征拼接融合→残差连接，捕捉双向上下文
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor):
        residual = x
        x_norm = self.norm(x)
        
        feat_fwd = self.mamba_fwd(x_norm)
        x_flipped = torch.flip(x_norm, dims=[1])
        feat_bwd = torch.flip(self.mamba_bwd(x_flipped), dims=[1])
        
        feat_concat = torch.cat([feat_fwd, feat_bwd], dim=-1)
        feat_fused = self.fusion(feat_concat)
        
        return feat_fused + residual

class CrossAttentionBridge(nn.Module):
    """
    层间跨云交互桥接模块
    在每个 BiMambaBlock 层之间插入，让 src 和 tgt 互相交流。
    src 作为 Query，attend to tgt（K,V）→ 更新 src
    tgt 作为 Query，attend to src（K,V）→ 更新 tgt
    两个方向使用独立的 cross_attention，学习不同的对应关系。
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # 独立的 LayerNorm，pre-norm 结构
        self.norm_src = nn.LayerNorm(d_model)
        self.norm_tgt = nn.LayerNorm(d_model)

        # src→tgt 和 tgt→src 使用独立的 attention，参数不共享
        # 这样两个方向能学到不同的对应关系模式
        self.cross_attn_src2tgt = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_tgt2src = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Args:
            src: [B, L, D]  当前层 src 特征
            tgt: [B, L, D]  当前层 tgt 特征
        Returns:
            src_out: [B, L, D]  交互后 src 特征
            tgt_out: [B, L, D]  交互后 tgt 特征
        """
        # pre-norm：先归一化，再计算 attention
        src_normed = self.norm_src(src)
        tgt_normed = self.norm_tgt(tgt)

        # src attend to tgt：Q=src, K=V=tgt
        # src 的每个 patch 去 tgt 中寻找对应的 patch
        src_cross, _ = self.cross_attn_src2tgt(src_normed, tgt_normed, tgt_normed)
        src_out = src + self.dropout(src_cross)  # 残差连接

        # tgt attend to src：Q=tgt, K=V=src
        # 注意：K,V 用的是 src_normed（交互前的值），保证两个方向的对称性
        tgt_cross, _ = self.cross_attn_tgt2src(tgt_normed, src_normed, src_normed)
        tgt_out = tgt + self.dropout(tgt_cross)  # 残差连接

        return src_out, tgt_out


class BiMambaBackbone(nn.Module):
    """双向 Mamba 骨干网络，含层间跨云交互桥接"""
    def __init__(self, d_model, n_layers=4, n_heads=8, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # 层间跨云交互桥：在前 n_layers-1 层之后各插一个
        # 最后一层之后已有 BATInteractionModule 做交互，无需重复
        self.cross_bridges = nn.ModuleList([
            CrossAttentionBridge(d_model, n_heads, dropout)
            for _ in range(n_layers - 1)   # 4层 → 3个桥
        ])

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Args:
            src: [B, L, D]  源点云 tokens
            tgt: [B, L, D]  目标点云 tokens
        Returns:
            src_out, tgt_out: [B, L, D]
            src_intermediate, tgt_intermediate: 各层输出列表（用于 DeepSupervision）
        """
        src_intermediate = []
        tgt_intermediate = []

        for i, layer in enumerate(self.layers):
            # 每层 Mamba 各自处理（保留 intra-cloud 序列建模能力）
            src = layer(src)
            tgt = layer(tgt)

            # 在前 n-1 层之后做跨云交互
            if i < len(self.cross_bridges):
                src, tgt = self.cross_bridges[i](src, tgt)

            src_intermediate.append(src)
            tgt_intermediate.append(tgt)

        src_out = self.final_norm(src)
        tgt_out = self.final_norm(tgt)

        return src_out, tgt_out, src_intermediate, tgt_intermediate


# ---------- Stage 3 & 4: BAT + Pose Decoder (引用前面的模块) ----------
# 这里为了完整性，将关键部分也包含进来

class BATInteractionModule(nn.Module):
    """BAT 交互模块 (简化版)"""
    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Cross Attention
        # 双向交叉注意力：源↔目标特征互增强；
        self.cross_attn_norm_src = nn.LayerNorm(d_model)
        self.cross_attn_norm_tgt = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # 位置编码
        # Patch 中心点编码融合到特征中，增强空间感知
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )
        
    def forward(self, src_coords, src_feat, tgt_coords, tgt_feat):
        B, N, C = src_feat.shape
        M = tgt_feat.shape[1]
        
        # 位置编码增强
        src_pos = self.pos_encoder(src_coords)
        tgt_pos = self.pos_encoder(tgt_coords)
        
        src_with_pos = src_feat + src_pos
        tgt_with_pos = tgt_feat + tgt_pos
        
        # Cross Attention: src -> tgt
        src_normed = self.cross_attn_norm_src(src_with_pos)
        tgt_normed = self.cross_attn_norm_tgt(tgt_with_pos)
        
        src_enhanced, _ = self.cross_attention(src_normed, tgt_normed, tgt_normed)
        src_enhanced = src_enhanced + src_feat
        
        tgt_enhanced, _ = self.cross_attention(tgt_normed, src_normed, src_normed)
        tgt_enhanced = tgt_enhanced + tgt_feat
        
        # Feature Replication & Broadcasting特征广播
        src_expanded = src_enhanced.unsqueeze(2).expand(B, N, M, C)
        tgt_expanded = tgt_enhanced.unsqueeze(1).expand(B, N, M, C)
        
        # Coordinate Info (10维)
        src_coord_exp = src_coords.unsqueeze(2).expand(B, N, M, 3)
        tgt_coord_exp = tgt_coords.unsqueeze(1).expand(B, N, M, 3)
        diff = src_coord_exp - tgt_coord_exp
        distance = torch.norm(diff, p=2, dim=-1, keepdim=True)
        geo_features = torch.cat([src_coord_exp, tgt_coord_exp, diff, distance], dim=-1)
        
        # Similarity (2维)余弦相似度 + 特征距离
        src_norm = F.normalize(src_enhanced, dim=-1)
        tgt_norm = F.normalize(tgt_enhanced, dim=-1)
        cosine_sim = (src_norm.unsqueeze(2) * tgt_norm.unsqueeze(1)).sum(dim=-1, keepdim=True)
        feat_dist = 1.0 / (1.0 + torch.norm(src_expanded - tgt_expanded, dim=-1, keepdim=True))
        similarity = torch.cat([cosine_sim, feat_dist], dim=-1)
        
        # Final Concatenation: [B, N, M, 2C+12]
        # 最终拼接：相似度 (2)+ 源特征 (D)+ 目标特征 (D)+ 几何特征 (10) → [B,L,L,2D+12]（L_bat）
        L_bat = torch.cat([similarity, src_expanded, tgt_expanded, geo_features], dim=-1)
        
        return L_bat, src_enhanced, tgt_enhanced


# =============================================================================
# 辅助函数：旋转矩阵 → 四元数 [x, y, z, w]（scalar-last，与 KITTI 数据集一致）
# =============================================================================

def _rot_mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    旋转矩阵批量转四元数（Shepperd 方法，全分支 batched，数值稳定）

    Args:
        R: 旋转矩阵 [B, 3, 3]

    Returns:
        q: 四元数 [B, 4]，格式 [x, y, z, w]（scalar-last），已 L2 归一化
    """
    B = R.shape[0]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # 分支 0: w 最大 (trace > 0)
    s0 = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2
    q0 = torch.stack([
        (R[:, 2, 1] - R[:, 1, 2]) / s0,
        (R[:, 0, 2] - R[:, 2, 0]) / s0,
        (R[:, 1, 0] - R[:, 0, 1]) / s0,
        0.25 * s0,
    ], dim=-1)

    # 分支 1: x 最大
    s1 = torch.sqrt(torch.clamp(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], min=1e-10)) * 2
    q1 = torch.stack([
        0.25 * s1,
        (R[:, 0, 1] + R[:, 1, 0]) / s1,
        (R[:, 0, 2] + R[:, 2, 0]) / s1,
        (R[:, 2, 1] - R[:, 1, 2]) / s1,
    ], dim=-1)

    # 分支 2: y 最大
    s2 = torch.sqrt(torch.clamp(1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2], min=1e-10)) * 2
    q2 = torch.stack([
        (R[:, 0, 1] + R[:, 1, 0]) / s2,
        0.25 * s2,
        (R[:, 1, 2] + R[:, 2, 1]) / s2,
        (R[:, 0, 2] - R[:, 2, 0]) / s2,
    ], dim=-1)

    # 分支 3: z 最大
    s3 = torch.sqrt(torch.clamp(1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1], min=1e-10)) * 2
    q3 = torch.stack([
        (R[:, 0, 2] + R[:, 2, 0]) / s3,
        (R[:, 1, 2] + R[:, 2, 1]) / s3,
        0.25 * s3,
        (R[:, 1, 0] - R[:, 0, 1]) / s3,
    ], dim=-1)

    cond1 = (trace > 0).unsqueeze(-1).expand_as(q0)
    cond2 = ((R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])).unsqueeze(-1).expand_as(q0)
    cond3 = (R[:, 1, 1] > R[:, 2, 2]).unsqueeze(-1).expand_as(q0)

    q = torch.where(cond1, q0,
        torch.where(cond2, q1,
        torch.where(cond3, q2, q3)))

    return F.normalize(q, p=2, dim=-1)


class BATPoseDecoder(nn.Module):
    """
    BAT 位姿解码器 v2 — 可微加权 SVD

    用 SVD 解析解替换原来的 MLP 旋转回归，彻底解除 SO(3) 学习天花板。

    流程:
        L_bat [B, N, M, 2C+12]
          -> corr_mlp    -> corr_weights   [B, N, M]  (softmax, 软对应权重)
          -> overlap_mlp -> overlap_scores [B, N, 1]  (sigmoid, 重叠置信度/点权重)
        + src_centroids  [B, N, 3]
        + tgt_centroids  [B, M, 3]
          -> WeightedSVD -> R [B,3,3] -> quaternion [B,4]
                        -> translation [B,3]

    梯度路径:
        QuaternionLoss(q)
          -> rot_mat_to_quat (closed-form, 可微)
          -> SVD (torch.linalg.svd, 已支持反传)
          -> H = f(corr_weights, overlap_scores, centroids)
          -> corr_mlp, overlap_mlp 收到梯度
    """
    def __init__(self, bat_feature_dim: int, global_feat_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        # global_feat_dim, dropout 保留以兼容旧 __init__ 调用，内部不再使用

        # 软对应评分 MLP: [B, N, M, D] -> [B, N, M, 1]
        self.corr_mlp = nn.Sequential(
            nn.Linear(bat_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # 重叠分数 MLP: 从聚合特征 [B, N, D] -> [B, N, 1]
        # 同时服务于：(a) SVD 点权重  (b) OverlapLoss GT 监督
        self.overlap_mlp = nn.Sequential(
            nn.Linear(bat_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        L_bat:          torch.Tensor,   # [B, N, M, 2C+12]
        src_centroids:  torch.Tensor,   # [B, N, 3]  src patch 中心点（来自 PatchEmbedding）
        tgt_centroids:  torch.Tensor,   # [B, M, 3]  tgt patch 中心点（来自 PatchEmbedding）
    ) -> Dict[str, torch.Tensor]:
        B, N, M, D = L_bat.shape
        device, dtype = L_bat.device, L_bat.dtype
        eps = 1e-8

        # ── Step 1: 软对应权重 ──────────────────────────────────────────
        corr_scores  = self.corr_mlp(L_bat).squeeze(-1)        # [B, N, M]
        corr_weights = F.softmax(corr_scores, dim=2)            # [B, N, M]

        # ── Step 2: 重叠分数（用聚合特征预测）──────────────────────────
        matched_features = (L_bat * corr_weights.unsqueeze(-1)).sum(dim=2)  # [B, N, D]
        overlap_scores   = self.overlap_mlp(matched_features)               # [B, N, 1]

        # ── Step 3: 软 tgt 位置 ─────────────────────────────────────────
        # 每个 src patch 对应的期望 tgt 位置
        # [B, N, M] @ [B, M, 3] -> [B, N, 3]
        tgt_matched = torch.bmm(corr_weights, tgt_centroids)

        # ── Step 4: 归一化点权重（overlap_scores → SVD 权重）────────────
        w = overlap_scores.squeeze(-1)                          # [B, N]
        w_norm = w / w.sum(dim=1, keepdim=True).clamp(min=eps) # [B, N]

        # ── Step 5: 加权质心 ─────────────────────────────────────────────
        src_bar = torch.einsum('bn,bnc->bc', w_norm, src_centroids)  # [B, 3]
        tgt_bar = torch.einsum('bn,bnc->bc', w_norm, tgt_matched)    # [B, 3]

        # ── Step 6: 中心化 ───────────────────────────────────────────────
        src_c = src_centroids - src_bar.unsqueeze(1)   # [B, N, 3]
        tgt_c = tgt_matched   - tgt_bar.unsqueeze(1)   # [B, N, 3]

        # ── Step 7: 加权叉积矩阵 H ──────────────────────────────────────
        w_exp = w_norm.unsqueeze(-1)                   # [B, N, 1]
        H = torch.bmm(
            (src_c * w_exp).transpose(1, 2),           # [B, 3, N]
            tgt_c,                                     # [B, N, 3]
        )                                              # [B, 3, 3]

        # ── Step 8: SVD + 反射修正 ───────────────────────────────────────
        try:
            U, S, Vh = torch.linalg.svd(H)            # U, Vh: [B, 3, 3]
        except RuntimeError:
            # SVD 不收敛（极罕见：全零点云等退化情况）
            R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
            t = tgt_bar - src_bar
            q = torch.zeros(B, 4, device=device, dtype=dtype)
            q[:, 3] = 1.0
            return {"quaternion": q, "translation": t, "rotation_matrix": R,
                    "correspondence_weights": corr_weights, "overlap_scores": overlap_scores}

        # 候选旋转
        R_raw = torch.bmm(Vh.transpose(-2, -1), U.transpose(-2, -1))  # [B, 3, 3]

        # 反射修正：使 det(R) = +1
        det_sign = torch.linalg.det(R_raw).sign()                     # [B]
        correction = torch.ones(B, 3, device=device, dtype=dtype)
        correction[:, 2] = det_sign
        Vh_fix = Vh * correction.unsqueeze(-1)                         # [B, 3, 3]
        R = torch.bmm(Vh_fix.transpose(-2, -1), U.transpose(-2, -1))  # [B, 3, 3]

        # ── Step 9: 平移 ─────────────────────────────────────────────────
        # t = tgt_bar - R @ src_bar
        translation = tgt_bar - torch.bmm(R, src_bar.unsqueeze(-1)).squeeze(-1)  # [B, 3]

        # ── Step 10: R → 四元数（用于 QuaternionLoss）───────────────────
        quaternion = _rot_mat_to_quat(R)                               # [B, 4]

        return {
            "quaternion":             quaternion,       # [B, 4]
            "translation":            translation,      # [B, 3]
            "rotation_matrix":        R,                # [B, 3, 3]（新增，调试用）
            "correspondence_weights": corr_weights,     # [B, N, M]
            "overlap_scores":         overlap_scores,   # [B, N, 1]
        }


# =============================================================================
# 完整的 RegMamba 模型
# =============================================================================

class RegMamba(nn.Module):
    """
    RegMamba: 基于 Z-Order + Bi-Mamba + BAT 的点云配准网络
    
    设计者：周女士
    
    完整流程:
        Stage 1: Input [B,N,3] -> PatchEmbedding -> tokens [B,L,D], centroids [B,L,3]
        Stage 2: tokens -> BiMambaBackbone -> features [B,L,D] + intermediate_feats
        Stage 3: features + centroids -> BATInteractionModule -> L_bat [B,L,L,2D+12]
        Stage 4: L_bat -> BATPoseDecoder -> quaternion [B,4], translation [B,3]
    """
    def __init__(self, config: RegMambaConfig):
        super().__init__()
        self.config = config
        
        print("=" * 70)
        print("  RegMamba 模型初始化")
        print("  设计者：周女士")
        print("=" * 70)
        print(f"\n{config}\n")
        
        # Stage 1: Patch Embedding
        self.patch_embedding = PatchEmbedding(config)
        
        # Stage 2: Bi-Directional Mamba Backbone
        self.backbone = BiMambaBackbone(
            d_model=config.d_model,
            n_layers=config.n_mamba_layers,
            n_heads=config.n_heads,       # ← 新增这一行
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            dropout=config.dropout,
        )
        
        # Stage 3: BAT Interaction Module
        self.bat_module = BATInteractionModule(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        
        # Stage 4: BAT Pose Decoder
        self.pose_decoder = BATPoseDecoder(
            bat_feature_dim=config.bat_feature_dim,
            global_feat_dim=512,
            dropout=config.dropout,
        )
        
        print(f"[RegMamba] 模型构建完成！")
        print(f"  - 序列长度 L: {config.seq_len}")
        print(f"  - BAT 特征维度: {config.bat_feature_dim}")
        
    def forward(
        self,
        src: torch.Tensor,  # [B, N, 3]
        tgt: torch.Tensor,  # [B, N, 3]
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            src: 源点云 [B, N, 3]
            tgt: 目标点云 [B, N, 3]
        
        Returns:
            output: Dict containing:
                - quaternion: 旋转四元数 [B, 4]
                - translation: 平移向量 [B, 3]
                - correspondence_weights: 对应权重 [B, L, L]
                - overlap_scores: 重叠分数 [B, L, 1]
                - src_intermediate_feats: 源中间层特征 List
                - tgt_intermediate_feats: 目标中间层特征 List
        """
        B, N, C = src.shape
        assert N == self.config.n_points, f"输入点数 {N} 与配置 {self.config.n_points} 不匹配"
        
        # ========== Stage 1: Patch Embedding ==========
        src_tokens, src_centroids, src_sort_idx = self.patch_embedding(src)
        tgt_tokens, tgt_centroids, tgt_sort_idx = self.patch_embedding(tgt)
        # src_tokens: [B, L, D], src_centroids: [B, L, 3]
        
        # ========== Stage 2: Bi-Mamba Backbone ==========
        # src_feat, src_intermediate = self.backbone(src_tokens)
        # tgt_feat, tgt_intermediate = self.backbone(tgt_tokens)
        # src_feat: [B, L, D]

        src_feat, tgt_feat, src_intermediate, tgt_intermediate = self.backbone(
        src_tokens, tgt_tokens)
        
        # ========== Stage 3: BAT Interaction Module ==========
        L_bat, src_enhanced, tgt_enhanced = self.bat_module(
            src_centroids, src_feat,
            tgt_centroids, tgt_feat,
        )
        # L_bat: [B, L, L, 2D+12]
        
        # ========== Stage 4: Pose Decoder (SVD) ==========
        # 新版 BATPoseDecoder 需要 src/tgt centroids 作为 SVD 的点坐标输入
        pose_output = self.pose_decoder(L_bat, src_centroids, tgt_centroids)
        # quaternion: [B, 4], translation: [B, 3]
        
        # ========== 组装输出 ==========
        '''
        输出包含：
            - quaternion: 旋转四元数 [B, 4]
            - translation: 平移向量 [B, 3]
            - correspondence_weights: 对应权重 [B, L, L]
            - overlap_scores: 重叠分数 [B, L, 1]
            - src_intermediate_feats: 源中间层特征 List
            - tgt_intermediate_feats: 目标中间层特征 List
            - src_centroids: 源 Patch 中心点 [B, L, 3]
            - tgt_centroids: 目标 Patch 中心点 [B, L, 3]
        '''
        output = {
            "quaternion": pose_output["quaternion"],
            "translation": pose_output["translation"],
            "correspondence_weights": pose_output["correspondence_weights"],
            "overlap_scores": pose_output["overlap_scores"],
            "src_intermediate_feats": src_intermediate,
            "tgt_intermediate_feats": tgt_intermediate,
            "src_centroids": src_centroids,
            "tgt_centroids": tgt_centroids,
        }
        
        return output
# =============================================================================
# 测试函数 (补全)
# =============================================================================

def test_regmamba():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    """测试完整的 RegMamba 模型"""
    print("\n" + "=" * 80)
    print("  测��� RegMamba 完整模型")
    print("  设计者：周女士")
    print("=" * 80 + "\n")
    
    torch.manual_seed(42)
    
    # 配置
    config = RegMambaConfig(
        n_points=14400,
        patch_size=32,
        stride=16,
        d_model=256,
        n_mamba_layers=4,
        n_heads=8,
    )
    
    # 创建模型
    model = RegMamba(config)
    model = model.to(device)
    # 测试数据
    B = 2  # 小 batch 测试
    src = torch.randn(B, config.n_points, 3)
    tgt = torch.randn(B, config.n_points, 3)
    
    print(f"输入形状: src={src.shape}, tgt={tgt.shape}")
    
    src = src.to(device)
    tgt = tgt.to(device)
    # 前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        output = model(src, tgt)
    
    # 检查输出
    print(f"\n输出形状:")
    print(f"  - quaternion: {output['quaternion'].shape}")
    print(f"  - translation: {output['translation'].shape}")
    print(f"  - correspondence_weights: {output['correspondence_weights'].shape}")
    print(f"  - overlap_scores: {output['overlap_scores'].shape}")
    print(f"  - 中间层数量: {len(output['src_intermediate_feats'])}")
    
    # 验证
    quat_norm = torch.norm(output['quaternion'], dim=-1)
    print(f"\n四元数范数: {quat_norm.mean().item():.6f} (期望: 1.0)")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")
    
    print("\n" + "=" * 80)
    print("RegMamba 测试完成！✓")
    print("=" * 80)
    
    return model, output


if __name__ == "__main__":
    model, output = test_regmamba()
