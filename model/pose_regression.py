"""
Pose Regression Module (BAT 之后的位姿回归)
================================================================================
设计者：周女士

从 BAT 输出的特征体 L_bat [B, N, M, 2C+12] 回归出位姿 (quaternion, translation)

流程:
    1. Soft Correspondence (软对应): 从 L_bat 中提取最佳匹配权重
    2. Weighted Feature Aggregation: 加权聚合得到全局特征
    3. Pose Regression Head: 回归四元数和平移向量
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional


# =============================================================================
# Soft Correspondence Module (软对应模块)
# =============================================================================

class SoftCorrespondence(nn.Module):
    """
    软对应模块
    
    从 L_bat [B, N, M, 2C+12] 中学习 Source 到 Target 的软对应关系
    
    输出:
        - correspondence_weights: 对应权重 [B, N, M]
        - overlap_scores: 重叠分数 [B, N] (每个源点是否在重叠区)
    """
    def __init__(self, in_channels: int, hidden_channels: int = 128):
        """
        Args:
            in_channels: 输入通道数 (2C + 12)
            hidden_channels: 隐藏层通道数
        """
        super().__init__()
        self.in_channels = in_channels
        
        # 对应关系评分网络: [B, N, M, 2C+12] -> [B, N, M, 1]
        self.correspondence_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 1),
        )
        
        # 重叠分数网络: 从聚合特征预测每个源点的重叠概率
        self.overlap_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self, 
        L_bat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            L_bat: BAT 特征体 [B, N, M, 2C+12]
        
        Returns:
            corr_weights: 软对应权重 [B, N, M] (每行和为1)
            overlap_scores: 重叠分数 [B, N, 1]
            matched_features: 匹配后的特征 [B, N, 2C+12]
        
        维度变化:
            L_bat [B, N, M, 2C+12] 
            -> correspondence_scores [B, N, M, 1] 
            -> softmax(dim=2) -> corr_weights [B, N, M]
            -> weighted sum -> matched_features [B, N, 2C+12]
        """
        B, N, M, D = L_bat.shape
        
        # ===== Step 1: 计算对应关系分数 =====
        # [B, N, M, 2C+12] -> [B, N, M, 1]
        corr_scores = self.correspondence_mlp(L_bat)  # [B, N, M, 1]
        corr_scores = corr_scores.squeeze(-1)         # [B, N, M]
        
        # Softmax 归一化 (每个源点对所有目标点的权重和为1)
        corr_weights = F.softmax(corr_scores, dim=2)  # [B, N, M]
        
        # ===== Step 2: 加权聚合特征 =====
        # [B, N, M] -> [B, N, M, 1] for broadcasting
        weights_expanded = corr_weights.unsqueeze(-1)  # [B, N, M, 1]
        
        # 加权求和: [B, N, M, D] * [B, N, M, 1] -> sum(dim=2) -> [B, N, D]
        matched_features = (L_bat * weights_expanded).sum(dim=2)  # [B, N, D]
        
        # ===== Step 3: 预测重叠分数 =====
        # 使用匹配后的特征预测每个源点是否在重叠区
        overlap_scores = self.overlap_mlp(matched_features)  # [B, N, 1]
        
        return corr_weights, overlap_scores, matched_features


# =============================================================================
# Global Feature Aggregation (全局特征聚合)
# =============================================================================

class GlobalFeatureAggregation(nn.Module):
    """
    全局特征聚合模块
    
    使用重叠分数加权聚合，得到用于位姿回归的全局特征
    
    公式: global_feat = sum(feat_i * score_i) / sum(score_i)
    """
    def __init__(self, in_channels: int, out_channels: int = 512):
        super().__init__()
        
        # 特征压缩: [B, N, 2C+12] -> [B, N, out_channels]
        self.feat_compress = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
        
        self.eps = 1e-8
        
    def forward(
        self,
        matched_features: torch.Tensor,  # [B, N, 2C+12]
        overlap_scores: torch.Tensor,     # [B, N, 1]
    ) -> torch.Tensor:
        """
        Args:
            matched_features: 匹配后的特征 [B, N, 2C+12]
            overlap_scores: 重叠分数 [B, N, 1]
        
        Returns:
            global_feat: 全局特征 [B, out_channels]
        
        维度变化:
            matched_features [B, N, 2C+12] -> compress -> [B, N, out_channels]
            weighted_sum: [B, N, out_channels] * [B, N, 1] -> sum -> [B, out_channels]
        """
        B, N, D = matched_features.shape
        
        # 特征压缩
        compressed = self.feat_compress(matched_features)  # [B, N, out_channels]
        
        # 加权聚合
        weighted_sum = (compressed * overlap_scores).sum(dim=1)  # [B, out_channels]
        score_sum = overlap_scores.sum(dim=1).clamp(min=self.eps)  # [B, 1]
        
        global_feat = weighted_sum / score_sum  # [B, out_channels]
        
        return global_feat


# =============================================================================
# Pose Regression Head (位姿回归头)
# =============================================================================

class PoseRegressionHead(nn.Module):
    """
    位姿回归头
    
    从全局特征回归旋转四元数和平移向量
    
    流程:
        global_feat [B, D] -> MLP (1024 -> 512 -> 256)
        分支1 (旋转): -> Linear(4) -> Normalize -> quaternion [B, 4]
        分支2 (平移): -> Linear(3) -> translation [B, 3]
    """
    def __init__(self, in_channels: int = 512, dropout: float = 0.1):
        super().__init__()
        
        # 共享 MLP
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        
        # 旋转分支 (四元数)
        self.quat_head = nn.Linear(256, 4)
        
        # 平移分支
        self.trans_head = nn.Linear(256, 3)
        
    def forward(self, global_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            global_feat: 全局特征 [B, in_channels]
        
        Returns:
            quaternion: 旋转四元数 [B, 4] (已归一化)
            translation: 平移向量 [B, 3]
        """
        # 共享 MLP
        feat = self.shared_mlp(global_feat)  # [B, 256]
        
        # 旋转分支
        quaternion = self.quat_head(feat)  # [B, 4]
        quaternion = F.normalize(quaternion, p=2, dim=-1)  # L2 归一化
        
        # 平移分支
        translation = self.trans_head(feat)  # [B, 3]
        
        return quaternion, translation


# =============================================================================
# 完整的 BAT Pose Decoder
# =============================================================================

class BATPoseDecoder(nn.Module):
    """
    BAT 位姿解码器
    
    将 BAT 模块输出的特征体 L_bat 解码为位姿
    
    完整流程:
        L_bat [B, N, M, 2C+12]
        -> SoftCorrespondence -> matched_features [B, N, 2C+12], overlap_scores [B, N, 1]
        -> GlobalFeatureAggregation -> global_feat [B, 512]
        -> PoseRegressionHead -> quaternion [B, 4], translation [B, 3]
    """
    def __init__(
        self,
        bat_feature_dim: int,  # 2C + 12
        global_feat_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bat_feature_dim = bat_feature_dim
        
        # 软对应模块
        self.soft_correspondence = SoftCorrespondence(
            in_channels=bat_feature_dim,
            hidden_channels=256,
        )
        
        # 全局特征聚合
        self.global_aggregation = GlobalFeatureAggregation(
            in_channels=bat_feature_dim,
            out_channels=global_feat_dim,
        )
        
        # 位姿回归头
        self.pose_head = PoseRegressionHead(
            in_channels=global_feat_dim,
            dropout=dropout,
        )
        
        print(f"[BATPoseDecoder] 初始化完成:")
        print(f"  - BAT 特征维度: {bat_feature_dim}")
        print(f"  - 全局特征维度: {global_feat_dim}")
        
    def forward(
        self, 
        L_bat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            L_bat: BAT 特征体 [B, N, M, 2C+12]
        
        Returns:
            output: Dict containing:
                - quaternion: 旋转四元数 [B, 4]
                - translation: 平移向量 [B, 3]
                - correspondence_weights: 对应权重 [B, N, M]
                - overlap_scores: 重叠分数 [B, N, 1]
        """
        # Step 1: 软对应
        corr_weights, overlap_scores, matched_features = self.soft_correspondence(L_bat)
        
        # Step 2: 全局特征聚合
        global_feat = self.global_aggregation(matched_features, overlap_scores)
        
        # Step 3: 位姿回归
        quaternion, translation = self.pose_head(global_feat)
        
        return {
            "quaternion": quaternion,
            "translation": translation,
            "correspondence_weights": corr_weights,
            "overlap_scores": overlap_scores,
        }


# =============================================================================
# 测试函数
# =============================================================================

def test_pose_decoder():
    """测试 BAT Pose Decoder"""
    print("=" * 80)
    print("测试 BAT Pose Decoder")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    # 参数
    B = 4
    N = 900
    M = 900
    C = 256
    bat_dim = 2 * C + 12  # 524
    
    # 模拟 BAT 输出
    L_bat = torch.randn(B, N, M, bat_dim)
    print(f"\n输入 L_bat: {L_bat.shape}")
    
    # 创建解码器
    decoder = BATPoseDecoder(
        bat_feature_dim=bat_dim,
        global_feat_dim=512,
        dropout=0.1,
    )
    
    # 前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        output = decoder(L_bat)
    
    # 检查输出
    print(f"\n输出:")
    print(f"  - quaternion: {output['quaternion'].shape}")
    print(f"  - translation: {output['translation'].shape}")
    print(f"  - correspondence_weights: {output['correspondence_weights'].shape}")
    print(f"  - overlap_scores: {output['overlap_scores'].shape}")
    
    # 验证
    quat_norm = torch.norm(output['quaternion'], dim=-1)
    print(f"\n四元数范数: {quat_norm.mean().item():.6f} (期望: 1.0)")
    
    corr_sum = output['correspondence_weights'].sum(dim=2)
    print(f"对应权重行��: {corr_sum.mean().item():.6f} (期望: 1.0)")
    
    print(f"重叠分数范围: [{output['overlap_scores'].min().item():.4f}, "
          f"{output['overlap_scores'].max().item():.4f}]")
    
    print("\n" + "=" * 80)
    print("BAT Pose Decoder 测试完成！✓")
    print("=" * 80)
    
    return decoder, output


if __name__ == "__main__":
    test_pose_decoder()