"""
Pose Regression Module (BAT 之后的位姿回归)
================================================================================
设计者：周女士

v2 改动：用可微 SVD 解析解替换原来的 MLP 四元数回归头。

原始流程（已废弃的 MLP 路径）:
    L_bat -> SoftCorrespondence -> GlobalAggregation -> MLP -> quaternion

新流程（SVD 解析解）:
    L_bat -> SoftCorrespondence
          -> corr_weights [B, L, M]   ← 软对应矩阵
          -> overlap_scores [B, L, 1] ← 重叠置信度（点权重）
    + src_centroids [B, L, 3]
    + tgt_centroids [B, M, 3]
    -> WeightedSVDSolver
          -> R [B, 3, 3]              ← 旋转矩阵（解析解）
          -> quaternion [B, 4]        ← 由 R 转换，用于损失计算
          -> translation [B, 3]       ← 解析解

为什么 SVD 优于 MLP 回归旋转：
    1. MLP 需要直接学习 SO(3) 流形上的非线性映射，梯度信号极弱，
       导致 Rotation Loss 在 Epoch 2 就平台化在 ~0.018（等效 ~22°）。
    2. SVD 是给定软对应点集后的最优旋转解析解（Procrustes 问题），
       不需要梯度爬坡，天然满足 SO(3) 约束（det=+1）。
    3. 梯度通过 corr_weights 和 overlap_scores 反传，驱动特征层改善对应质量。
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


# =============================================================================
# 辅助函数：旋转矩阵 → 四元数 [x, y, z, w]（scalar-last，与 KITTI 数据集一致）
# =============================================================================

def rot_mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    旋转矩阵批量转四元数（Shepperd 方法，全分支 batched，数值稳定）

    Args:
        R: 旋转矩阵 [B, 3, 3]

    Returns:
        q: 四元数 [B, 4]，格式 [x, y, z, w]（scalar-last），已 L2 归一化

    四元数约定与 train.py / compute_patch_correspondence 保持一致。
    """
    B = R.shape[0]

    # ---- 预计算 4 个分支的 s（避免分支内 sqrt(负数) ）----
    # 每个分支对应四元数中最大分量的情形
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]          # [B]

    # 分支 0: w 最大  (trace > 0)
    s0 = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2  # 4w
    q0 = torch.stack([
        (R[:, 2, 1] - R[:, 1, 2]) / s0,   # x
        (R[:, 0, 2] - R[:, 2, 0]) / s0,   # y
        (R[:, 1, 0] - R[:, 0, 1]) / s0,   # z
        0.25 * s0,                          # w
    ], dim=-1)  # [B, 4]

    # 分支 1: x 最大  (R00 > R11, R00 > R22)
    s1 = torch.sqrt(torch.clamp(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], min=1e-10)) * 2  # 4x
    q1 = torch.stack([
        0.25 * s1,                          # x
        (R[:, 0, 1] + R[:, 1, 0]) / s1,   # y
        (R[:, 0, 2] + R[:, 2, 0]) / s1,   # z
        (R[:, 2, 1] - R[:, 1, 2]) / s1,   # w
    ], dim=-1)

    # 分支 2: y 最大  (R11 > R22)
    s2 = torch.sqrt(torch.clamp(1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2], min=1e-10)) * 2  # 4y
    q2 = torch.stack([
        (R[:, 0, 1] + R[:, 1, 0]) / s2,   # x
        0.25 * s2,                          # y
        (R[:, 1, 2] + R[:, 2, 1]) / s2,   # z
        (R[:, 0, 2] - R[:, 2, 0]) / s2,   # w
    ], dim=-1)

    # 分支 3: z 最大
    s3 = torch.sqrt(torch.clamp(1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1], min=1e-10)) * 2  # 4z
    q3 = torch.stack([
        (R[:, 0, 2] + R[:, 2, 0]) / s3,   # x
        (R[:, 1, 2] + R[:, 2, 1]) / s3,   # y
        0.25 * s3,                          # z
        (R[:, 1, 0] - R[:, 0, 1]) / s3,   # w
    ], dim=-1)

    # ---- 按 Shepperd 条件选择最优分支（避免分母过小导致数值爆炸）----
    cond1 = (trace > 0).unsqueeze(-1).expand_as(q0)
    cond2 = ((R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])).unsqueeze(-1).expand_as(q0)
    cond3 = (R[:, 1, 1] > R[:, 2, 2]).unsqueeze(-1).expand_as(q0)

    q = torch.where(cond1, q0,
        torch.where(cond2, q1,
        torch.where(cond3, q2, q3)))

    return F.normalize(q, p=2, dim=-1)


# =============================================================================
# 软对应模块（保持不变，同时服务于 SVD 权重和 OverlapLoss 监督）
# =============================================================================

class SoftCorrespondence(nn.Module):
    """
    软对应模块

    从 L_bat [B, N, M, 2C+12] 中学习：
        - corr_weights   [B, N, M]：src patch → tgt patch 的软对应权重（softmax）
        - overlap_scores [B, N, 1]：每个 src patch 是否在重叠区的置信度（0~1）

    这两个输出同时：
        (a) 送入 WeightedSVDSolver 作为位姿估计的输入权重
        (b) 由 OverlapLoss 对 overlap_scores 进行 GT 监督，保证其语义正确
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256):
        super().__init__()

        # 对应关系评分 MLP: [B, N, M, D] -> [B, N, M, 1]
        self.correspondence_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 1),
        )

        # 重叠分数 MLP: 从聚合后特征 [B, N, D] -> [B, N, 1]
        self.overlap_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, L_bat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            L_bat: BAT 特征体 [B, N, M, 2C+12]

        Returns:
            corr_weights:     软对应权重 [B, N, M]（每行和为 1）
            overlap_scores:   重叠分数 [B, N, 1]
            matched_features: corr_weights 加权聚合后的特征 [B, N, 2C+12]
        """
        B, N, M, D = L_bat.shape

        # 对应关系评分 & softmax 归一化
        corr_scores  = self.correspondence_mlp(L_bat).squeeze(-1)  # [B, N, M]
        corr_weights = F.softmax(corr_scores, dim=2)                # [B, N, M]

        # 软聚合：每个 src patch 对应的期望 tgt 特征
        matched_features = (L_bat * corr_weights.unsqueeze(-1)).sum(dim=2)  # [B, N, D]

        # 重叠分数
        overlap_scores = self.overlap_mlp(matched_features)  # [B, N, 1]

        return corr_weights, overlap_scores, matched_features


# =============================================================================
# 核心新增：可微 SVD 位姿求解器
# =============================================================================

class WeightedSVDSolver(nn.Module):
    """
    可微加权 SVD 位姿求解器（Weighted Procrustes Problem）

    输入：
        corr_weights   [B, L, M]  — 软对应矩阵（每行和为 1）
        overlap_scores [B, L, 1]  — 重叠置信度（用作点权重）
        src_centroids  [B, L, 3]  — src patch 中心点
        tgt_centroids  [B, M, 3]  — tgt patch 中心点

    算法步骤：
        1. 软 tgt 位置：tgt_matched = corr_weights @ tgt_centroids  → [B, L, 3]
           （每个 src patch 对应的期望 tgt 位置，通过对应权重加权平均得到）
        2. 归一化权重：w = overlap_scores / sum(overlap_scores)
        3. 加权质心：src_bar, tgt_bar
        4. 中心化点云
        5. 叉积矩阵 H = src_centered^T @ diag(w) @ tgt_centered  → [B, 3, 3]
        6. SVD(H) = U @ S @ Vh
        7. R = Vh^T @ U^T，处理 det=-1 的反射情况
        8. t = tgt_bar - R @ src_bar
        9. R → 四元数（用于 QuaternionLoss）

    梯度流：
        loss(q, t)
          ↓ 通过 rot_mat_to_quat（closed-form，可微）
        loss(R)
          ↓ 通过 SVD（torch.linalg.svd，已实现可微反传）
        loss(H = f(corr_weights, overlap_scores, src/tgt_centroids))
          ↓
        corr_weights, overlap_scores 收到梯度，驱动特征层改善对应质量
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        corr_weights:   torch.Tensor,   # [B, L, M]
        overlap_scores: torch.Tensor,   # [B, L, 1]
        src_centroids:  torch.Tensor,   # [B, L, 3]
        tgt_centroids:  torch.Tensor,   # [B, M, 3]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            quaternion:  [B, 4]    旋转四元数 [x,y,z,w]（已归一化，用于损失）
            translation: [B, 3]    平移向量
            R:           [B, 3, 3] 旋转矩阵（输出给下游，如可视化）
        """
        B, L, M = corr_weights.shape
        device, dtype = corr_weights.device, corr_weights.dtype

        # ------------------------------------------------------------------
        # Step 1: 软对应 → 每个 src patch 对应的期望 tgt 位置
        # corr_weights [B, L, M] @ tgt_centroids [B, M, 3] → [B, L, 3]
        # 含义：第 i 个 src patch 对应的 tgt 位置 = Σ_j w_ij * tgt_j
        # ------------------------------------------------------------------
        tgt_matched = torch.bmm(corr_weights, tgt_centroids)  # [B, L, 3]

        # ------------------------------------------------------------------
        # Step 2: 归一化点权重（overlap_scores → 每个 src patch 的信任度）
        # 只有重叠区域的 patch 才参与旋转估计，非重叠 patch 权重接近 0
        # ------------------------------------------------------------------
        w = overlap_scores.squeeze(-1)                                       # [B, L]
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=self.eps)               # [B, 1]
        w_norm = w / w_sum                                                    # [B, L]，归一化

        # ------------------------------------------------------------------
        # Step 3: 加权质心
        # einsum 'bl,blc->bc': Σ_i w_i * point_i，结果 [B, 3]
        # ------------------------------------------------------------------
        src_bar = torch.einsum('bl,blc->bc', w_norm, src_centroids)         # [B, 3]
        tgt_bar = torch.einsum('bl,blc->bc', w_norm, tgt_matched)           # [B, 3]

        # ------------------------------------------------------------------
        # Step 4: 中心化
        # ------------------------------------------------------------------
        src_centered = src_centroids - src_bar.unsqueeze(1)                 # [B, L, 3]
        tgt_centered = tgt_matched   - tgt_bar.unsqueeze(1)                 # [B, L, 3]

        # ------------------------------------------------------------------
        # Step 5: 加权叉积矩阵  H = X^T W Y
        # 等价于: H = (w_norm * src_centered)^T @ tgt_centered
        # [B, 3, L] @ [B, L, 3] → [B, 3, 3]
        # ------------------------------------------------------------------
        w_expanded = w_norm.unsqueeze(-1)                                    # [B, L, 1]
        H = torch.bmm(
            (src_centered * w_expanded).transpose(1, 2),                    # [B, 3, L]
            tgt_centered,                                                    # [B, L, 3]
        )                                                                    # [B, 3, 3]

        # ------------------------------------------------------------------
        # Step 6 & 7: SVD + 反射修正
        # R = Vh^T @ diag_correction @ U^T
        # ------------------------------------------------------------------
        try:
            U, S, Vh = torch.linalg.svd(H)                                  # U,Vh: [B,3,3]
        except RuntimeError:
            # SVD 不收敛（极罕见，通常是全零点云）：退化为单位旋转
            R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
            t = tgt_bar - src_bar
            q = torch.zeros(B, 4, device=device, dtype=dtype)
            q[:, 3] = 1.0   # w = 1 → 单位旋转
            return q, t, R

        # 候选旋转矩阵
        R_raw = torch.bmm(Vh.transpose(-2, -1), U.transpose(-2, -1))       # [B, 3, 3]

        # 检测反射（det = -1 时为反射，不是合法旋转）
        # 修正：将 Vh 最后一行乘以 det 的符号，使 det(R) = +1
        det = torch.linalg.det(R_raw)                                       # [B]
        # 构造修正对角矩阵 [1, 1, sign(det)]，广播到 Vh 每一列
        det_sign = det.sign()                                                # [B]
        correction = torch.ones(B, 3, device=device, dtype=dtype)
        correction[:, 2] = det_sign
        Vh_corrected = Vh * correction.unsqueeze(-1)                        # [B, 3, 3]
        R = torch.bmm(Vh_corrected.transpose(-2, -1), U.transpose(-2, -1)) # [B, 3, 3]

        # ------------------------------------------------------------------
        # Step 8: 平移向量
        # t = tgt_bar - R @ src_bar
        # [B, 3, 3] @ [B, 3, 1] → squeeze → [B, 3]
        # ------------------------------------------------------------------
        t = tgt_bar - torch.bmm(R, src_bar.unsqueeze(-1)).squeeze(-1)      # [B, 3]

        # ------------------------------------------------------------------
        # Step 9: 旋转矩阵 → 四元数（用于 QuaternionLoss）
        # ------------------------------------------------------------------
        q = rot_mat_to_quat(R)                                               # [B, 4]

        return q, t, R


# =============================================================================
# 完整的 BAT Pose Decoder（v2：SVD 版本）
# =============================================================================

class BATPoseDecoder(nn.Module):
    """
    BAT 位姿解码器 v2（可微 SVD）

    流程:
        L_bat [B, N, M, 2C+12]
          -> SoftCorrespondence
               -> corr_weights   [B, N, M]
               -> overlap_scores [B, N, 1]
          + src_centroids [B, N, 3]
          + tgt_centroids [B, M, 3]
          -> WeightedSVDSolver
               -> quaternion [B, 4]   （解析解，不经过 MLP）
               -> translation [B, 3]   （解析解）
               -> R [B, 3, 3]
    """
    def __init__(
        self,
        bat_feature_dim: int,           # 2C + 12
        global_feat_dim: int = 512,     # 保留参数接口，不再使用（兼容旧调用代码）
        dropout: float = 0.1,           # 保留参数接口，不再使用
    ):
        super().__init__()
        self.bat_feature_dim = bat_feature_dim

        # 软对应模块（保留，同时为 SVD 提供权重 & 为 OverlapLoss 提供监督信号）
        self.soft_correspondence = SoftCorrespondence(
            in_channels=bat_feature_dim,
            hidden_channels=256,
        )

        # SVD 求解器（无可学习参数，纯解析计算）
        self.svd_solver = WeightedSVDSolver(eps=1e-8)

        print(f"[BATPoseDecoder v2 - SVD] 初始化完成:")
        print(f"  - BAT 特征维度: {bat_feature_dim}")
        print(f"  - 位姿求解: 可微加权 SVD（解析解，替代原 MLP 旋转回归）")

    def forward(
        self,
        L_bat:          torch.Tensor,   # [B, N, M, 2C+12]
        src_centroids:  torch.Tensor,   # [B, N, 3]  src patch 中心点
        tgt_centroids:  torch.Tensor,   # [B, M, 3]  tgt patch 中心点
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            L_bat:         BAT 特征体 [B, N, M, 2C+12]
            src_centroids: src patch 中心点 [B, N, 3]  ← 新增，来自 PatchEmbedding
            tgt_centroids: tgt patch 中心点 [B, M, 3]  ← 新增，来自 PatchEmbedding

        Returns:
            quaternion:             旋转四元数 [B, 4]（SVD 解析解 → rot_mat_to_quat）
            translation:            平移向量 [B, 3]（SVD 解析解）
            rotation_matrix:        旋转矩阵 [B, 3, 3]（调试/可视化用）
            correspondence_weights: 软对应权重 [B, N, M]
            overlap_scores:         重叠分数 [B, N, 1]
        """
        # Step 1: 软对应
        corr_weights, overlap_scores, _ = self.soft_correspondence(L_bat)
        # corr_weights:   [B, N, M]
        # overlap_scores: [B, N, 1]

        # Step 2: SVD 解析解
        quaternion, translation, R = self.svd_solver(
            corr_weights=corr_weights,
            overlap_scores=overlap_scores,
            src_centroids=src_centroids,
            tgt_centroids=tgt_centroids,
        )

        return {
            "quaternion":             quaternion,       # [B, 4]
            "translation":            translation,      # [B, 3]
            "rotation_matrix":        R,                # [B, 3, 3]（新增输出）
            "correspondence_weights": corr_weights,     # [B, N, M]
            "overlap_scores":         overlap_scores,   # [B, N, 1]
        }


# =============================================================================
# 测试函数
# =============================================================================

def test_pose_decoder():
    """测试 BATPoseDecoder v2（SVD 版本）"""
    print("=" * 80)
    print("测试 BATPoseDecoder v2 — 可微 SVD")
    print("=" * 80)

    torch.manual_seed(42)

    B, N, M, C = 4, 640, 640, 128
    bat_dim = 2 * C + 12   # 268

    # 模拟输入
    L_bat         = torch.randn(B, N, M, bat_dim)
    src_centroids = torch.randn(B, N, 3) * 10   # 模拟真实点云尺度
    tgt_centroids = torch.randn(B, M, 3) * 10

    print(f"\n输入: L_bat={L_bat.shape}, src={src_centroids.shape}, tgt={tgt_centroids.shape}")

    decoder = BATPoseDecoder(bat_feature_dim=bat_dim)

    print("\n执行前向传播...")
    with torch.no_grad():
        output = decoder(L_bat, src_centroids, tgt_centroids)

    print(f"\n输出:")
    for k, v in output.items():
        print(f"  {k}: {v.shape}")

    # 验证约束
    q = output['quaternion']
    R = output['rotation_matrix']
    print(f"\n验证:")
    print(f"  四元数范数:    {torch.norm(q, dim=-1).mean():.6f}  (期望: 1.0)")
    print(f"  det(R):        {torch.linalg.det(R).mean():.6f}   (期望: 1.0)")
    print(f"  R^T R 误差:    {(torch.bmm(R.transpose(-2,-1), R) - torch.eye(3)).abs().max():.2e}  (期望: ~0)")
    print(f"  对应权重行和:  {output['correspondence_weights'].sum(dim=2).mean():.6f}  (期望: 1.0)")
    print(f"  重叠分数范围:  [{output['overlap_scores'].min():.4f}, {output['overlap_scores'].max():.4f}]")

    # 验证梯度可以反传
    output2 = decoder(L_bat.requires_grad_(False), src_centroids, tgt_centroids)
    fake_loss = output2['quaternion'].sum() + output2['translation'].sum()
    fake_loss.backward()
    print(f"\n  梯度反传: ✓  (corr_mlp.weight.grad is not None: "
          f"{decoder.soft_correspondence.correspondence_mlp[0].weight.grad is not None})")

    print("\n" + "=" * 80)
    print("BATPoseDecoder v2 测试完成！✓")
    print("=" * 80)


if __name__ == "__main__":
    test_pose_decoder()
