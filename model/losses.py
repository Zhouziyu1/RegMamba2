"""
Loss Functions for RegMamba
================================================================================
设计者：周女士

损失函数组成:
    L_total = λ1 * L_rot + λ2 * L_trans + λ3 * L_overlap + λ_ds * Σ L_feat^l

1. L_rot: 旋转损失 (四元数距离)
2. L_trans: 平移损失 (L2 距离)
3. L_overlap: 重叠区监督损失 (可选，需要 GT mask)
4. L_feat: 逐层深度监督损失 (Circle Loss / InfoNCE)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# =============================================================================
# 1. 旋转损失 (Quaternion Loss)
# =============================================================================

class QuaternionLoss(nn.Module):
    """
    四元数旋转损失
    
    公式: L_rot = 1 - |q · q_gt|
    
    注意：四元数 q 和 -q 表示相同的旋转，所以取绝对值
    """
    def __init__(self):
        super().__init__()
        
    def forward(
        self, 
        pred_q: torch.Tensor,  # [B, 4]
        gt_q: torch.Tensor,    # [B, 4]
    ) -> torch.Tensor:
        """
        Args:
            pred_q: 预测四元数 [B, 4] (已归一化)
            gt_q: 真实四元数 [B, 4] (已归一化)
        
        Returns:
            loss: 旋转损失 (标量)
        """
        # 确保输入已归一化
        pred_q = F.normalize(pred_q, p=2, dim=-1)
        gt_q = F.normalize(gt_q, p=2, dim=-1)
        
        # 计算四元数点积
        dot_product = (pred_q * gt_q).sum(dim=-1)  # [B]
        
        # 取绝对值 (处理 q 和 -q 等价性)
        dot_product = torch.abs(dot_product)
        
        # 损失: 1 - |q · q_gt|
        loss = 1.0 - dot_product  # [B]
        
        return loss.mean()


# =============================================================================
# 2. 平移损失 (Translation Loss)
# =============================================================================

class TranslationLoss(nn.Module):
    """
    平移向量损失
    
    公式: L_trans = ||t - t_gt||_2 (MSE)
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        
    def forward(
        self,
        pred_t: torch.Tensor,  # [B, 3]
        gt_t: torch.Tensor,    # [B, 3]
    ) -> torch.Tensor:
        """
        Args:
            pred_t: 预测平移向量 [B, 3]
            gt_t: 真实平移向量 [B, 3]
        
        Returns:
            loss: 平移损失 (标量)
        """
        return self.mse(pred_t, gt_t)


# =============================================================================
# 3. 重叠区监督损失 (Overlap Loss)
# =============================================================================

class OverlapLoss(nn.Module):
    """
    重叠区监督损失
    
    如果数据集中有重叠区的 Mask (如 3DMatch)，则监督预测的 overlap_scores
    
    公式: L_overlap = BCE(scores, gt_mask)
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction='mean')
        
    def forward(
        self,
        pred_scores: torch.Tensor,  # [B, N, 1] or [B, N]
        gt_mask: torch.Tensor,      # [B, N] (0/1 二值 mask)
    ) -> torch.Tensor:
        """
        Args:
            pred_scores: 预测的重叠分数 [B, N, 1] or [B, N]
            gt_mask: 真实的重叠 mask [B, N]
        
        Returns:
            loss: 重叠损失 (标量)
        """
        # 调整维度
        if pred_scores.dim() == 3:
            pred_scores = pred_scores.squeeze(-1)  # [B, N]
        
        gt_mask = gt_mask.float()
        
        return self.bce(pred_scores, gt_mask)


# =============================================================================
# 4. 深度监督特征损失 (Feature Loss with Circle Loss / InfoNCE)
# =============================================================================

class CircleLoss(nn.Module):
    """
    Circle Loss for Deep Supervision
    
    用于逐层 Mamba 特征对齐
    
    参考: Circle Loss: A Unified Perspective of Pair Similarity Optimization
    """
    def __init__(
        self,
        m: float = 0.25,      # margin
        gamma: float = 80.0,  # scale factor
    ):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()
        
    def forward(
        self,
        src_feat: torch.Tensor,  # [B, N, D]
        tgt_feat: torch.Tensor,  # [B, M, D]
        correspondence: torch.Tensor,  # [B, N] 对应的目标点索引
    ) -> torch.Tensor:
        """
        Args:
            src_feat: 源特征 [B, N, D]
            tgt_feat: 目标特征 [B, M, D]
            correspondence: 对应关系 [B, N] (每个源点对应的目标点索引)
        
        Returns:
            loss: Circle Loss (标量)
        """
        B, N, D = src_feat.shape
        M = tgt_feat.shape[1]
        
        # 归一化特征
        src_feat = F.normalize(src_feat, p=2, dim=-1)
        tgt_feat = F.normalize(tgt_feat, p=2, dim=-1)
        
        # 计算所有点对的相似度: [B, N, M]
        similarity = torch.bmm(src_feat, tgt_feat.transpose(1, 2))
        
        # 获取正样本相似度
        batch_idx = torch.arange(B, device=src_feat.device).view(B, 1).expand(B, N)
        src_idx = torch.arange(N, device=src_feat.device).view(1, N).expand(B, N)
        pos_sim = similarity[batch_idx, src_idx, correspondence]  # [B, N]
        
        # 计算负样本相似度 (排除正样本)
        mask = torch.ones(B, N, M, device=src_feat.device)
        mask[batch_idx, src_idx, correspondence] = 0
        neg_sim = similarity * mask  # [B, N, M]
        
        # Circle Loss 计算
        # 正样本权重
        ap = torch.clamp_min(-pos_sim.detach() + 1 + self.m, min=0.)
        # 负样本权重
        an = torch.clamp_min(neg_sim.detach() + self.m, min=0.)
        
        # Delta
        delta_p = 1 - self.m
        delta_n = self.m
        
        # LogSumExp
        logit_p = -ap * (pos_sim - delta_p) * self.gamma
        logit_n = an * (neg_sim - delta_n) * self.gamma
        
        loss = self.soft_plus(
            torch.logsumexp(logit_n, dim=2) + torch.logsumexp(-logit_p.unsqueeze(2), dim=2)
        )
        
        return loss.mean()


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for Deep Supervision
    
    对比学习���失，拉近正样本对，推远负样本对
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        src_feat: torch.Tensor,  # [B, N, D]
        tgt_feat: torch.Tensor,  # [B, M, D]
        correspondence: Optional[torch.Tensor] = None,  # [B, N]
    ) -> torch.Tensor:
        """
        Args:
            src_feat: 源特征 [B, N, D]
            tgt_feat: 目标特征 [B, M, D]
            correspondence: 对应关系 [B, N] (可选，如果没有则使用对角线)
        
        Returns:
            loss: InfoNCE Loss (标量)
        """
        B, N, D = src_feat.shape
        M = tgt_feat.shape[1]
        
        # 归一化
        src_feat = F.normalize(src_feat, p=2, dim=-1)
        tgt_feat = F.normalize(tgt_feat, p=2, dim=-1)
        
        # 计算相似度矩阵: [B, N, M]
        similarity = torch.bmm(src_feat, tgt_feat.transpose(1, 2)) / self.temperature
        
        if correspondence is None:
            # 假设对角线是正样本 (N == M)
            assert N == M, "没有对应关系时，N 必须等于 M"
            labels = torch.arange(N, device=src_feat.device).unsqueeze(0).expand(B, -1)
        else:
            labels = correspondence  # [B, N]
        
        # Cross Entropy Loss
        # similarity: [B, N, M] -> [B*N, M]
        # labels: [B, N] -> [B*N]
        similarity_flat = similarity.reshape(B * N, M)
        labels_flat = labels.reshape(B * N)
        
        loss = F.cross_entropy(similarity_flat, labels_flat)
        
        return loss


class DeepSupervisionLoss(nn.Module):
    """
    深度监督损失
    
    对 Mamba Backbone 的每一层输出计算特征对齐损失
    
    公式: L_feat = Σ_{l=1}^{L} L_feat^l
    """
    def __init__(
        self,
        loss_type: str = 'infonce',  # 'circle' or 'infonce'
        temperature: float = 0.07,
        circle_m: float = 0.25,
        circle_gamma: float = 80.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'circle':
            self.loss_fn = CircleLoss(m=circle_m, gamma=circle_gamma)
        else:
            self.loss_fn = InfoNCELoss(temperature=temperature)
            
    def forward(
        self,
        src_intermediate: List[torch.Tensor],  # List of [B, L, D]
        tgt_intermediate: List[torch.Tensor],  # List of [B, L, D]
        correspondence: Optional[torch.Tensor] = None,  # [B, L]
    ) -> torch.Tensor:
        """
        Args:
            src_intermediate: 源中间层特征列表 [layer1, layer2, ...]
            tgt_intermediate: 目标中间层特征列表
            correspondence: 对应关系 (可选)
        
        Returns:
            loss: 深度监督损失 (标量)
        """
        assert len(src_intermediate) == len(tgt_intermediate), \
            "源和目标的中间层数量必须相同"
        
        total_loss = 0.0
        n_layers = len(src_intermediate)
        
        for src_feat, tgt_feat in zip(src_intermediate, tgt_intermediate):
            if self.loss_type == 'circle' and correspondence is not None:
                layer_loss = self.loss_fn(src_feat, tgt_feat, correspondence)
            else:
                layer_loss = self.loss_fn(src_feat, tgt_feat, correspondence)
            total_loss += layer_loss
        
        # 平均
        return total_loss / n_layers


# =============================================================================
# 5. 完整损失函数
# =============================================================================

class RegMambaLoss(nn.Module):
    """
    RegMamba 完整损失函数
    设计者：周女士
    
    L_total = λ1 * L_rot + λ2 * L_trans + λ3 * L_overlap + λ_ds * Σ L_feat^l
    """
    
    def __init__(
        self,
        lambda_rot: float = 1.0,
        lambda_trans: float = 1.0,
        lambda_overlap: float = 0.5,
        lambda_ds: float = 0.1,
        use_deep_supervision: bool = True,
        use_overlap_loss: bool = False,
        ds_loss_type: str = 'infonce',
    ):
        super().__init__()
        
        self.lambda_rot = lambda_rot
        self.lambda_trans = lambda_trans
        self.lambda_overlap = lambda_overlap
        self.lambda_ds = lambda_ds
        self.use_deep_supervision = use_deep_supervision
        self.use_overlap_loss = use_overlap_loss
        
        # 子损失函数
        self.rot_loss = QuaternionLoss()
        self.trans_loss = TranslationLoss()
        
        if use_overlap_loss:
            self.overlap_loss = OverlapLoss()
        
        if use_deep_supervision:
            self.ds_loss = DeepSupervisionLoss(loss_type=ds_loss_type)
            
        print(f"[RegMambaLoss] 初始化完成:")
        print(f"  - λ_rot: {lambda_rot}")
        print(f"  - λ_trans: {lambda_trans}")
        print(f"  - λ_overlap: {lambda_overlap if use_overlap_loss else 'N/A (disabled)'}")
        print(f"  - λ_ds: {lambda_ds if use_deep_supervision else 'N/A (disabled)'}")
    
    def forward(
        self,
        pred_q: torch.Tensor,
        pred_t: torch.Tensor,
        gt_q: torch.Tensor,
        gt_t: torch.Tensor,
        overlap_scores: Optional[torch.Tensor] = None,
        overlap_gt: Optional[torch.Tensor] = None,
        src_intermediate: Optional[List[torch.Tensor]] = None,
        tgt_intermediate: Optional[List[torch.Tensor]] = None,
        correspondence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            pred_q: 预测四元数 [B, 4]
            pred_t: 预测平移 [B, 3]
            gt_q: 真实四元数 [B, 4]
            gt_t: 真实平移 [B, 3]
            overlap_scores: 预测重叠分数 [B, N, 1] (可选)
            overlap_gt: 真实重叠mask [B, N] (可选)
            src_intermediate: 源中间层特征 List[[B, L, D]] (可选)
            tgt_intermediate: 目标中间层特征 List[[B, L, D]] (可选)
            correspondence: 点对应关系 (可选)
        
        Returns:
            loss_dict: Dict containing:
                - total_loss: 总损失
                - rot_loss: 旋转损失
                - trans_loss: 平移损失
                - overlap_loss: 重叠损失
                - ds_loss: 深度监督总损失
                - feature_losses: 逐层特征损失列表 [layer1_loss, layer2_loss, ...]
        """
        loss_dict = {}
        total_loss = 0.0
        
        # ===== 1. 旋转损失 L_rot =====
        l_rot = self.rot_loss(pred_q, gt_q)
        loss_dict['rot_loss'] = l_rot
        total_loss += self.lambda_rot * l_rot
        
        # ===== 2. 平移损失 L_trans =====
        l_trans = self.trans_loss(pred_t, gt_t)
        loss_dict['trans_loss'] = l_trans
        total_loss += self.lambda_trans * l_trans
        
        # ===== 3. 重叠损失 L_overlap (可选) =====
        if self.use_overlap_loss and overlap_scores is not None and overlap_gt is not None:
            l_overlap = self.overlap_loss(overlap_scores, overlap_gt)
            loss_dict['overlap_loss'] = l_overlap
            total_loss += self.lambda_overlap * l_overlap
        else:
            loss_dict['overlap_loss'] = torch.tensor(0.0, device=pred_q.device)
        
        # ===== 4. 深度监督损失 L_ds (逐层计算并记录) =====
        feature_losses = []
        
        if self.use_deep_supervision and src_intermediate is not None and tgt_intermediate is not None:
            n_layers = min(len(src_intermediate), len(tgt_intermediate))
            
            for layer_idx in range(n_layers):
                src_feat = src_intermediate[layer_idx]
                tgt_feat = tgt_intermediate[layer_idx]
                
                # 计算该层的特征对齐损失
                if hasattr(self.ds_loss, 'loss_fn'):
                    layer_loss = self.ds_loss.loss_fn(src_feat, tgt_feat, correspondence)
                else:
                    # 简化版：使用余弦相似度损失
                    src_norm = F.normalize(src_feat, dim=-1)
                    tgt_norm = F.normalize(tgt_feat, dim=-1)
                    sim = torch.bmm(src_norm, tgt_norm.transpose(1, 2))
                    # 对角线应该最大
                    L = src_feat.shape[1]
                    target = torch.arange(L, device=src_feat.device).unsqueeze(0).expand(src_feat.shape[0], -1)
                    layer_loss = F.cross_entropy(sim.view(-1, L), target.view(-1))
                
                feature_losses.append(layer_loss)
            
            # 总深度监督损失 = 各层平均
            if len(feature_losses) > 0:
                l_ds = sum(feature_losses) / len(feature_losses)
                loss_dict['ds_loss'] = l_ds
                total_loss += self.lambda_ds * l_ds
            else:
                loss_dict['ds_loss'] = torch.tensor(0.0, device=pred_q.device)
        else:
            loss_dict['ds_loss'] = torch.tensor(0.0, device=pred_q.device)
        
        # ===== 记录逐层特征损失 (用于Excel记录) =====
        loss_dict['feature_losses'] = feature_losses
        
        # ===== 总损失 =====
        loss_dict['total_loss'] = total_loss
        
        return loss_dict


# =============================================================================
# 测试函数
# =============================================================================

def test_losses():
    """测试损失函数"""
    print("=" * 80)
    print("测试损失函数")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    B = 4
    N = 900
    D = 256
    n_layers = 4
    
    # 模拟数据
    pred_q = F.normalize(torch.randn(B, 4), dim=-1)
    pred_t = torch.randn(B, 3)
    gt_q = F.normalize(torch.randn(B, 4), dim=-1)
    gt_t = torch.randn(B, 3)
    
    overlap_scores = torch.sigmoid(torch.randn(B, N, 1))
    overlap_gt = (torch.rand(B, N) > 0.5).float()
    
    src_intermediate = [torch.randn(B, N, D) for _ in range(n_layers)]
    tgt_intermediate = [torch.randn(B, N, D) for _ in range(n_layers)]
    
    # 创建损失函数
    criterion = RegMambaLoss(
        lambda_rot=1.0,
        lambda_trans=1.0,
        lambda_overlap=0.5,
        lambda_ds=0.1,
        use_deep_supervision=True,
        use_overlap_loss=True,
    )
    
    # 计算损失
    print("\n计算损失...")
    loss_dict = criterion(
        pred_q=pred_q,
        pred_t=pred_t,
        gt_q=gt_q,
        gt_t=gt_t,
        overlap_scores=overlap_scores,
        overlap_gt=overlap_gt,
        src_intermediate=src_intermediate,
        tgt_intermediate=tgt_intermediate,
    )
    
    # 打印结果
    print("\n损失结果:")
    for name, value in loss_dict.items():
        print(f"  - {name}: {value.item():.6f}")
    
    print("\n" + "=" * 80)
    print("损失函数测试完成！✓")
    print("=" * 80)
    
    return criterion, loss_dict


if __name__ == "__main__":
    test_losses()