"""
BAT (Bi-directional All-to-all Transformer) Interaction Module
================================================================================
设计者：周女士

BAT 模块用于在 Mamba Backbone 之后，构建 Source 和 Target 之间的全对全交互。

输入:
    - Source: 坐标 P_S [B, N, 3], 特征 F_S [B, N, C]
    - Target: 坐标 P_T [B, M, 3], 特征 F_T [B, M, C]
    - (注: 在模型中 N=M ≈ 900)

输出:
    - L_bat: All-to-all 特征体 [B, N, M, 2C+12]

步骤:
    1. Cross Attention (位置编码增强)
    2. Feature Replication (构造 N×M 网格)
    3. Coordinate Information Extraction (10维几何特征)
    4. Cosine Similarity (相似度计算, 2维)
    5. Final Concatenation
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


# =============================================================================
# Step 1: Cross Attention with Positional Encoding
# =============================================================================

class PositionalEncodingLayer(nn.Module):
    """
    位置编码层
    
    将3D坐标编码为高维特征，用于增强Cross Attention
    """
    def __init__(self, d_model: int = 256, d_pos: int = 64):
        super().__init__()
        self.d_pos = d_pos
        
        # 坐标编码 MLP: 3 -> d_pos
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, d_pos),
            nn.ReLU(inplace=True),
            nn.Linear(d_pos, d_pos),
        )
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: 坐标 [B, N, 3]
        
        Returns:
            pos_encoding: 位置编码 [B, N, d_pos]
        """
        return self.pos_encoder(coords)


class CrossAttentionWithPE(nn.Module):
    """
    带位置编码增强的 Cross Attention
    
    输入 Source 和 Target 的特征与坐标，通过标准的 Multi-head Cross Attention 进行上下文交互。
    
    输出: 增强后的特征 F̃_S (对应 cf_i) 和 F̃_T (对应 cf_j)
    形状保持不变: [B, N, C] 和 [B, M, C]
    """
    def __init__(
        self, 
        d_model: int = 256, 
        n_heads: int = 8, 
        d_pos: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, f"d_model({d_model})必须能被n_heads({n_heads})整除"
        
        self.scale = self.head_dim ** -0.5
        
        # 位置编码
        self.pos_encoder = PositionalEncodingLayer(d_model, d_pos)
        
        # 特征+位置 融合
        self.feat_pos_fusion = nn.Linear(d_model + d_pos, d_model)
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer Normalization
        self.norm_src = nn.LayerNorm(d_model)
        self.norm_tgt = nn.LayerNorm(d_model)
        
        # FFN for residual
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        
    def _cross_attention(
        self, 
        query_feat: torch.Tensor,      # [B, N, C]
        query_coords: torch.Tensor,    # [B, N, 3]
        key_feat: torch.Tensor,        # [B, M, C]
        key_coords: torch.Tensor,      # [B, M, 3]
    ) -> torch.Tensor:
        """
        单向 Cross Attention: Query attends to Key/Value
        
        Returns:
            attended_feat: [B, N, C]
        """
        B, N, C = query_feat.shape
        M = key_feat.shape[1]
        
        # 位置编码
        query_pos = self.pos_encoder(query_coords)  # [B, N, d_pos]
        key_pos = self.pos_encoder(key_coords)      # [B, M, d_pos]
        
        # 特征+位置融合
        query_fused = self.feat_pos_fusion(
            torch.cat([query_feat, query_pos], dim=-1)
        )  # [B, N, C]
        key_fused = self.feat_pos_fusion(
            torch.cat([key_feat, key_pos], dim=-1)
        )  # [B, M, C]
        
        # Q, K, V 投影
        Q = self.q_proj(query_fused)  # [B, N, C]
        K = self.k_proj(key_fused)    # [B, M, C]
        V = self.v_proj(key_feat)     # [B, M, C]  注意: V 使用原始特征
        
        # Multi-head reshape
        Q = Q.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, dim]
        K = K.reshape(B, M, self.n_heads, self.head_dim).transpose(1, 2)  # [B, heads, M, dim]
        V = V.reshape(B, M, self.n_heads, self.head_dim).transpose(1, 2)  # [B, heads, M, dim]
        
        # Attention scores: [B, heads, N, M]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Attend to values: [B, heads, N, dim]
        attended = torch.matmul(attn_probs, V)
        
        # Reshape back: [B, N, C]
        attended = attended.transpose(1, 2).reshape(B, N, C)
        
        return attended
        
    def forward(
        self,
        src_feat: torch.Tensor,     # [B, N, C]
        src_coords: torch.Tensor,   # [B, N, 3]
        tgt_feat: torch.Tensor,     # [B, M, C]
        tgt_coords: torch.Tensor,   # [B, M, 3]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        双向 Cross Attention
        
        Args:
            src_feat: 源特征 [B, N, C]
            src_coords: 源坐标 [B, N, 3]
            tgt_feat: 目标特征 [B, M, C]
            tgt_coords: 目标坐标 [B, M, 3]
        
        Returns:
            src_enhanced: 增强后的源特征 F̃_S [B, N, C]
            tgt_enhanced: 增强后的目标特征 F̃_T [B, M, C]
        
        维度变化:
            输入/输出形状保持不变！
        """
        # ===== Source -> Target Cross Attention =====
        # Source attends to Target
        src_normed = self.norm_src(src_feat)
        tgt_normed = self.norm_tgt(tgt_feat)
        
        src_attended = self._cross_attention(
            query_feat=src_normed,
            query_coords=src_coords,
            key_feat=tgt_normed,
            key_coords=tgt_coords,
        )  # [B, N, C]
        src_attended = self.out_proj(src_attended)
        src_enhanced = src_feat + src_attended  # Residual
        
        # FFN for Source
        src_enhanced = src_enhanced + self.ffn(self.norm_ffn(src_enhanced))
        
        # ===== Target -> Source Cross Attention =====
        # Target attends to Source
        tgt_attended = self._cross_attention(
            query_feat=tgt_normed,
            query_coords=tgt_coords,
            key_feat=src_normed,
            key_coords=src_coords,
        )  # [B, M, C]
        tgt_attended = self.out_proj(tgt_attended)
        tgt_enhanced = tgt_feat + tgt_attended  # Residual
        
        # FFN for Target
        tgt_enhanced = tgt_enhanced + self.ffn(self.norm_ffn(tgt_enhanced))
        
        return src_enhanced, tgt_enhanced


# =============================================================================
# Step 2: Feature Replication (构造 N×M 网格)
# =============================================================================

class FeatureReplication(nn.Module):
    """
    特征复制模块：将特征扩展为 N×M 网格
    
    F̃_S^{expand} ∈ [B, N, 1, C]
    F̃_T^{expand} ∈ [B, 1, M, C]
    
    通过广播机制，准备生成 N×M 的点对
    """
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        src_feat: torch.Tensor,  # [B, N, C]
        tgt_feat: torch.Tensor,  # [B, M, C]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src_feat: 源特征 [B, N, C]
            tgt_feat: 目标特征 [B, M, C]
        
        Returns:
            src_expanded: 扩展的源特征 [B, N, 1, C]
            tgt_expanded: 扩展的目标特征 [B, 1, M, C]
        
        注意：后续通过广播机制可以得到 [B, N, M, C] 的配对
        """
        # 在不同维度上 unsqueeze
        src_expanded = src_feat.unsqueeze(2)  # [B, N, C] -> [B, N, 1, C]
        tgt_expanded = tgt_feat.unsqueeze(1)  # [B, M, C] -> [B, 1, M, C]
        
        return src_expanded, tgt_expanded


# =============================================================================
# Step 3: Coordinate Information Extraction (10维几何特征)
# =============================================================================

class CoordinateInfoExtractor(nn.Module):
    """
    坐标信息提取模块 (纯几何计算，不依赖网络权重)
    
    对于每一对点 (p_i, p_j)，构建 10 维向量 r_ij:
        - p_i (Source 坐标): 3 维
        - p_j (Target 坐标): 3 维
        - p_i - p_j (差向量): 3 维
        - ||p_i - p_j|| (欧氏距离): 1 维
    
    输出: R_geo ∈ [B, N, M, 10]
    """
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        src_coords: torch.Tensor,  # [B, N, 3]
        tgt_coords: torch.Tensor,  # [B, M, 3]
    ) -> torch.Tensor:
        """
        Args:
            src_coords: 源坐标 [B, N, 3]
            tgt_coords: 目标坐标 [B, M, 3]
        
        Returns:
            geo_features: 几何特征 [B, N, M, 10]
        
        维度变化:
            src_coords: [B, N, 3] -> [B, N, 1, 3] (expand)
            tgt_coords: [B, M, 3] -> [B, 1, M, 3] (expand)
            通过广播得到 [B, N, M, 10]
        """
        B, N, _ = src_coords.shape
        M = tgt_coords.shape[1]
        
        # 扩展坐标用于广播
        src_expanded = src_coords.unsqueeze(2)  # [B, N, 1, 3]
        tgt_expanded = tgt_coords.unsqueeze(1)  # [B, 1, M, 3]
        
        # 通过广播扩展到 [B, N, M, 3]
        src_broadcast = src_expanded.expand(B, N, M, 3)  # p_i
        tgt_broadcast = tgt_expanded.expand(B, N, M, 3)  # p_j
        
        # 计算差向量
        diff = src_broadcast - tgt_broadcast  # p_i - p_j: [B, N, M, 3]
        
        # 计算欧氏距离
        distance = torch.norm(diff, p=2, dim=-1, keepdim=True)  # ||p_i - p_j||: [B, N, M, 1]
        
        # 拼接 10 维几何特征
        geo_features = torch.cat([
            src_broadcast,   # p_i: 3维
            tgt_broadcast,   # p_j: 3维
            diff,            # p_i - p_j: 3维
            distance,        # ||p_i - p_j||: 1维
        ], dim=-1)  # [B, N, M, 10]
        
        return geo_features


# =============================================================================
# Step 4: Cosine Similarity (相似度计算)
# =============================================================================

class SimilarityComputation(nn.Module):
    """
    相似度计算模块
    
    计算 F̃_S 和 F̃_T 之间的特征相似度
    输出: S_map �� [B, N, M, 2]
    
    包含:
        1. Cosine Similarity (余弦相似度)
        2. Feature Distance (特征距离，经过归一化)
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(
        self,
        src_feat: torch.Tensor,  # [B, N, C]
        tgt_feat: torch.Tensor,  # [B, M, C]
    ) -> torch.Tensor:
        """
        Args:
            src_feat: 源特征 [B, N, C]
            tgt_feat: 目标特征 [B, M, C]
        
        Returns:
            similarity_map: 相似度图 [B, N, M, 2]
        
        维度变化:
            src_feat: [B, N, C] -> [B, N, 1, C]
            tgt_feat: [B, M, C] -> [B, 1, M, C]
            cosine_sim: [B, N, M, 1]
            feat_dist: [B, N, M, 1]
            concat -> [B, N, M, 2]
        """
        B, N, C = src_feat.shape
        M = tgt_feat.shape[1]
        
        # 扩展特征用于广播
        src_expanded = src_feat.unsqueeze(2)  # [B, N, 1, C]
        tgt_expanded = tgt_feat.unsqueeze(1)  # [B, 1, M, C]
        
        # 归一化特征 (用于 cosine similarity)
        src_normalized = F.normalize(src_expanded, p=2, dim=-1)  # [B, N, 1, C]
        tgt_normalized = F.normalize(tgt_expanded, p=2, dim=-1)  # [B, 1, M, C]
        
        # Cosine Similarity: 点积后得到 [B, N, M, 1]
        # 注意：需要先广播再计算
        cosine_sim = (src_normalized * tgt_normalized).sum(dim=-1, keepdim=True)  # [B, N, M, 1]
        
        # Feature L2 Distance (归一化到 [0, 1] 范围)
        feat_diff = src_expanded - tgt_expanded  # [B, N, M, C]
        feat_dist = torch.norm(feat_diff, p=2, dim=-1, keepdim=True)  # [B, N, M, 1]
        
        # 归一化距离 (使用 sigmoid 或者 min-max normalization)
        # 这里使用 1 / (1 + dist) 归一化，距离越近值越大
        feat_dist_normalized = 1.0 / (1.0 + feat_dist)  # [B, N, M, 1]
        
        # 拼接得到 2 维相似度特征
        similarity_map = torch.cat([
            cosine_sim,           # 余弦相似度: 1维
            feat_dist_normalized, # 归一化距离: 1维
        ], dim=-1)  # [B, N, M, 2]
        
        return similarity_map


# =============================================================================
# Step 5: BAT Interaction Module (完整模块)
# =============================================================================

class BATInteractionModule(nn.Module):
    """
    BAT (Bi-directional All-to-all Transformer) Interaction Module
    
    完整流程:
        1. Cross Attention (位置编码增强) -> F̃_S, F̃_T [B, N, C], [B, M, C]
        2. Feature Replication -> [B, N, 1, C], [B, 1, M, C]
        3. Coordinate Info Extraction -> R_geo [B, N, M, 10]
        4. Cosine Similarity -> S_map [B, N, M, 2]
        5. Final Concatenation -> L_bat [B, N, M, 2C+12]
    
    输入:
        - Source: 坐标 P_S [B, N, 3], 特征 F_S [B, N, C]
        - Target: 坐标 P_T [B, M, 3], 特征 F_T [B, M, C]
    
    输出:
        - L_bat: All-to-all 特征体 [B, N, M, 2C+12]
        - F̃_S: 增强后的源特征 [B, N, C]
        - F̃_T: 增强后的目标特征 [B, M, C]
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_pos: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Step 1: Cross Attention with Positional Encoding
        self.cross_attention = CrossAttentionWithPE(
            d_model=d_model,
            n_heads=n_heads,
            d_pos=d_pos,
            dropout=dropout,
        )
        
        # Step 2: Feature Replication
        self.feature_replication = FeatureReplication()
        
        # Step 3: Coordinate Info Extraction
        self.coord_extractor = CoordinateInfoExtractor()
        
        # Step 4: Similarity Computation
        self.similarity_computation = SimilarityComputation()
        
        print(f"[BATInteractionModule] 初始化完成:")
        print(f"  - d_model (C): {d_model}")
        print(f"  - 输出维度: 2C + 12 = {2 * d_model + 12}")
        
    def forward(
        self,
        src_coords: torch.Tensor,   # P_S: [B, N, 3]
        src_feat: torch.Tensor,     # F_S: [B, N, C]
        tgt_coords: torch.Tensor,   # P_T: [B, M, 3]
        tgt_feat: torch.Tensor,     # F_T: [B, M, C]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        BAT 模块前向传播
        
        Args:
            src_coords: 源坐标 [B, N, 3]
            src_feat: 源特征 [B, N, C]
            tgt_coords: 目标坐标 [B, M, 3]
            tgt_feat: 目标特征 [B, M, C]
        
        Returns:
            L_bat: All-to-all 特征体 [B, N, M, 2C+12]
            src_enhanced: 增强后的源特征 F̃_S [B, N, C]
            tgt_enhanced: 增强后的目标特征 F̃_T [B, M, C]
        
        详细维度变化:
            Step 1: Cross Attention
                F_S [B, N, C] + F_T [B, M, C] -> F̃_S [B, N, C], F̃_T [B, M, C]
            
            Step 2: Feature Replication
                F̃_S [B, N, C] -> F̃_S_expand [B, N, 1, C]
                F̃_T [B, M, C] -> F̃_T_expand [B, 1, M, C]
            
            Step 3: Coordinate Info
                P_S [B, N, 3] + P_T [B, M, 3] -> R_geo [B, N, M, 10]
            
            Step 4: Similarity
                F̃_S [B, N, C] + F̃_T [B, M, C] -> S_map [B, N, M, 2]
            
            Step 5: Concatenation
                S_map [B, N, M, 2]
                + F̃_S_expand (broadcast) [B, N, M, C]
                + F̃_T_expand (broadcast) [B, N, M, C]
                + R_geo [B, N, M, 10]
                = L_bat [B, N, M, 2C+12]
        """
        B, N, C = src_feat.shape
        M = tgt_feat.shape[1]
        
        # ========== Step 1: Cross Attention (位置编码增强) ==========
        src_enhanced, tgt_enhanced = self.cross_attention(
            src_feat=src_feat,
            src_coords=src_coords,
            tgt_feat=tgt_feat,
            tgt_coords=tgt_coords,
        )  # F̃_S: [B, N, C], F̃_T: [B, M, C]
        
        # ========== Step 2: Feature Replication ==========
        src_expanded, tgt_expanded = self.feature_replication(
            src_enhanced, tgt_enhanced
        )  # [B, N, 1, C], [B, 1, M, C]
        
        # 广播到 [B, N, M, C]
        src_broadcast = src_expanded.expand(B, N, M, C)  # [B, N, M, C]
        tgt_broadcast = tgt_expanded.expand(B, N, M, C)  # [B, N, M, C]
        
        # ========== Step 3: Coordinate Info Extraction (10维几何特征) ==========
        geo_features = self.coord_extractor(
            src_coords, tgt_coords
        )  # R_geo: [B, N, M, 10]
        
        # ========== Step 4: Cosine Similarity (2维) ==========
        similarity_map = self.similarity_computation(
            src_enhanced, tgt_enhanced
        )  # S_map: [B, N, M, 2]
        
        # ========== Step 5: Final Concatenation ==========
        # L_ij = Concat(S_map, F̃_S_expand, F̃_T_expand, R_geo)
        # 总维度: 2 + C + C + 10 = 2C + 12
        L_bat = torch.cat([
            similarity_map,   # [B, N, M, 2]
            src_broadcast,    # [B, N, M, C]
            tgt_broadcast,    # [B, N, M, C]
            geo_features,     # [B, N, M, 10]
        ], dim=-1)  # [B, N, M, 2C+12]
        
        return L_bat, src_enhanced, tgt_enhanced


# =============================================================================
# 测试函数
# =============================================================================

def test_bat_module():
    """测试 BAT Interaction Module"""
    print("=" * 80)
    print("测试 BAT Interaction Module")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    # 参数设置
    B = 4       # Batch size
    N = 900     # Source 点数 (序列长度 L)
    M = 900     # Target 点数
    C = 256     # 特征维度
    
    # 创建测试数据
    src_coords = torch.randn(B, N, 3)   # P_S
    src_feat = torch.randn(B, N, C)     # F_S
    tgt_coords = torch.randn(B, M, 3)   # P_T
    tgt_feat = torch.randn(B, M, C)     # F_T
    
    print(f"\n输入维度:")
    print(f"  - src_coords (P_S): {src_coords.shape}")
    print(f"  - src_feat (F_S): {src_feat.shape}")
    print(f"  - tgt_coords (P_T): {tgt_coords.shape}")
    print(f"  - tgt_feat (F_T): {tgt_feat.shape}")
    
    # 创建 BAT 模块
    bat_module = BATInteractionModule(
        d_model=C,
        n_heads=8,
        d_pos=64,
        dropout=0.1,
    )
    
    # 前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        L_bat, src_enhanced, tgt_enhanced = bat_module(
            src_coords, src_feat, tgt_coords, tgt_feat
        )
    
    # 检查输出
    expected_dim = 2 * C + 12  # 2C + 12 = 524
    print(f"\n输出维度:")
    print(f"  - L_bat: {L_bat.shape}")
    print(f"    期望: [B={B}, N={N}, M={M}, 2C+12={expected_dim}]")
    print(f"    实际最后一维: {L_bat.shape[-1]} (期望: {expected_dim})")
    print(f"  - src_enhanced (F̃_S): {src_enhanced.shape}")
    print(f"  - tgt_enhanced (F̃_T): {tgt_enhanced.shape}")
    
    # 验证维度
    assert L_bat.shape == (B, N, M, expected_dim), \
        f"L_bat 维度错误: 期望 {(B, N, M, expected_dim)}, 实际 {L_bat.shape}"
    assert src_enhanced.shape == (B, N, C), \
        f"src_enhanced 维度错误: 期望 {(B, N, C)}, 实际 {src_enhanced.shape}"
    assert tgt_enhanced.shape == (B, M, C), \
        f"tgt_enhanced 维度错误: 期望 {(B, M, C)}, 实际 {tgt_enhanced.shape}"
    
    # 显存估计 (仅供参考)
    l_bat_memory_mb = L_bat.numel() * 4 / (1024 ** 2)  # float32, MB
    print(f"\n显存估计:")
    print(f"  - L_bat 占用: {l_bat_memory_mb:.2f} MB")
    print(f"  - (N×M = {N}×{M} = {N*M:,} 点对)")
    
    # 统计参数量
    total_params = sum(p.numel() for p in bat_module.parameters())
    print(f"\nBAT 模块参数量: {total_params:,}")
    
    print("\n" + "=" * 80)
    print("BAT Interaction Module 测试完成！✓")
    print("=" * 80)
    
    return bat_module, L_bat


# =============================================================================
# 分步测试函数
# =============================================================================

def test_bat_steps():
    """分步测试 BAT 模块的每个组件"""
    print("=" * 80)
    print("分步测试 BAT 模块")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    B, N, M, C = 2, 100, 100, 64  # 小规模测试
    
    src_coords = torch.randn(B, N, 3)
    src_feat = torch.randn(B, N, C)
    tgt_coords = torch.randn(B, M, 3)
    tgt_feat = torch.randn(B, M, C)
    
    # Step 1: Cross Attention
    print("\nStep 1: Cross Attention with PE")
    cross_attn = CrossAttentionWithPE(d_model=C, n_heads=4, d_pos=32)
    src_enhanced, tgt_enhanced = cross_attn(src_feat, src_coords, tgt_feat, tgt_coords)
    print(f"  输出: src_enhanced {src_enhanced.shape}, tgt_enhanced {tgt_enhanced.shape}")
    
    # Step 2: Feature Replication
    print("\nStep 2: Feature Replication")
    replication = FeatureReplication()
    src_exp, tgt_exp = replication(src_enhanced, tgt_enhanced)
    print(f"  输出: src_expanded {src_exp.shape}, tgt_expanded {tgt_exp.shape}")
    
    # Step 3: Coordinate Info
    print("\nStep 3: Coordinate Info Extraction (10维)")
    coord_extractor = CoordinateInfoExtractor()
    geo_features = coord_extractor(src_coords, tgt_coords)
    print(f"  输出: R_geo {geo_features.shape}")
    
    # Step 4: Similarity
    print("\nStep 4: Similarity Computation (2维)")
    sim_module = SimilarityComputation()
    sim_map = sim_module(src_enhanced, tgt_enhanced)
    print(f"  输出: S_map {sim_map.shape}")
    
    # Step 5: Final Concatenation
    print("\nStep 5: Final Concatenation")
    src_broadcast = src_exp.expand(B, N, M, C)
    tgt_broadcast = tgt_exp.expand(B, N, M, C)
    L_bat = torch.cat([sim_map, src_broadcast, tgt_broadcast, geo_features], dim=-1)
    print(f"  输出: L_bat {L_bat.shape}")
    print(f"  维度分解: 2 + {C} + {C} + 10 = {2 + C + C + 10}")
    
    print("\n" + "=" * 80)
    print("分步测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    # 分步测试
    test_bat_steps()
    
    print("\n" + "=" * 80 + "\n")
    
    # 完整测试
    bat_module, L_bat = test_bat_module()