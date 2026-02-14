# -*- coding:UTF-8 -*-
"""
点云配准可视化工具
================================================================================
设计者：周女士

功能:
1. 点云可视化 (源、目标、配准结果)
2. 误差分布直方图
3. 召回率曲线
4. 对应关系可视化
5. Attention Map 可视化
6. 训练曲线可视化
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from typing import Dict, List, Optional, Tuple
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 1. 点云可视化
# =============================================================================

def visualize_point_clouds(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    transformed_src: np.ndarray = None,
    title: str = "点云配准可视化",
    save_path: str = None,
    point_size: float = 0.5,
    figsize: Tuple[int, int] = (15, 5),
    elev: float = 30,
    azim: float = 45,
):
    """
    可视化点云配准结果
    
    Args:
        src_points: 源点云 (N, 3)
        tgt_points: 目标点云 (M, 3)
        transformed_src: 变换后的源点云 (N, 3)，可选
        title: 图标题
        save_path: 保存路径
        point_size: 点大小
        figsize: 图尺寸
    """
    n_cols = 3 if transformed_src is not None else 2
    fig = plt.figure(figsize=figsize)
    
    # 子图1: 源点云
    ax1 = fig.add_subplot(1, n_cols, 1, projection='3d')
    ax1.scatter(src_points[:, 0], src_points[:, 1], src_points[:, 2],
                c='blue', s=point_size, alpha=0.6, label='Source')
    ax1.set_title('源点云 (Source)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=elev, azim=azim)
    
    # 子图2: 目标点云
    ax2 = fig.add_subplot(1, n_cols, 2, projection='3d')
    ax2.scatter(tgt_points[:, 0], tgt_points[:, 1], tgt_points[:, 2],
                c='green', s=point_size, alpha=0.6, label='Target')
    ax2.set_title('目标点云 (Target)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=elev, azim=azim)
    
    # 子图3: 配准结果
    if transformed_src is not None:
        ax3 = fig.add_subplot(1, n_cols, 3, projection='3d')
        ax3.scatter(transformed_src[:, 0], transformed_src[:, 1], transformed_src[:, 2],
                    c='red', s=point_size, alpha=0.6, label='Transformed Source')
        ax3.scatter(tgt_points[:, 0], tgt_points[:, 1], tgt_points[:, 2],
                    c='green', s=point_size, alpha=0.4, label='Target')
        ax3.set_title('配准结果 (Aligned)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()
        ax3.view_init(elev=elev, azim=azim)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[可视化] 已保存: {save_path}")
    
    plt.show()
    plt.close()


def visualize_registration_comparison(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    pred_transformed: np.ndarray,
    gt_transformed: np.ndarray,
    title: str = "配准对比",
    save_path: str = None,
):
    """
    对比预测配准和真实配准
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 预测结果
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(pred_transformed[:, 0], pred_transformed[:, 1], pred_transformed[:, 2],
                c='red', s=0.5, alpha=0.6, label='Predicted')
    ax1.scatter(tgt_points[:, 0], tgt_points[:, 1], tgt_points[:, 2],
                c='green', s=0.5, alpha=0.4, label='Target')
    ax1.set_title('预测配准 (Prediction)')
    ax1.legend()
    
    # 真实结果
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(gt_transformed[:, 0], gt_transformed[:, 1], gt_transformed[:, 2],
                c='blue', s=0.5, alpha=0.6, label='Ground Truth')
    ax2.scatter(tgt_points[:, 0], tgt_points[:, 1], tgt_points[:, 2],
                c='green', s=0.5, alpha=0.4, label='Target')
    ax2.set_title('真实配准 (Ground Truth)')
    ax2.legend()
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    plt.close()


# =============================================================================
# 2. 误差分布可视化
# =============================================================================

def visualize_error_distribution(
    rot_errors: np.ndarray,
    trans_errors: np.ndarray,
    rot_thresh: float = 5.0,
    trans_thresh: float = 2.0,
    title: str = "误差分布",
    save_path: str = None,
):
    """
    可视化旋转和平移误差分布
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 旋转误差直方图
    ax1 = axes[0, 0]
    ax1.hist(rot_errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(rot_thresh, color='red', linestyle='--', linewidth=2, label=f'阈值={rot_thresh}°')
    ax1.set_xlabel('旋转误差 (°)')
    ax1.set_ylabel('样本数')
    ax1.set_title('旋转误差分布')
    ax1.legend()
    
    # 平移误差直方图
    ax2 = axes[0, 1]
    ax2.hist(trans_errors, bins=50, color='seagreen', edgecolor='black', alpha=0.7)
    ax2.axvline(trans_thresh, color='red', linestyle='--', linewidth=2, label=f'阈值={trans_thresh}m')
    ax2.set_xlabel('平移误差 (m)')
    ax2.set_ylabel('样本数')
    ax2.set_title('平移误差分布')
    ax2.legend()
    
    # 旋转误差累积分布
    ax3 = axes[1, 0]
    sorted_rot = np.sort(rot_errors)
    cdf_rot = np.arange(1, len(sorted_rot) + 1) / len(sorted_rot) * 100
    ax3.plot(sorted_rot, cdf_rot, color='steelblue', linewidth=2)
    ax3.axvline(rot_thresh, color='red', linestyle='--', linewidth=2)
    ax3.axhline(np.mean(rot_errors < rot_thresh) * 100, color='orange', linestyle=':', linewidth=2)
    ax3.set_xlabel('旋转误差 (°)')
    ax3.set_ylabel('累积百分比 (%)')
    ax3.set_title('旋转误差累积分布 (CDF)')
    ax3.grid(True, alpha=0.3)
    
    # 平移误差累积分布
    ax4 = axes[1, 1]
    sorted_trans = np.sort(trans_errors)
    cdf_trans = np.arange(1, len(sorted_trans) + 1) / len(sorted_trans) * 100
    ax4.plot(sorted_trans, cdf_trans, color='seagreen', linewidth=2)
    ax4.axvline(trans_thresh, color='red', linestyle='--', linewidth=2)
    ax4.axhline(np.mean(trans_errors < trans_thresh) * 100, color='orange', linestyle=':', linewidth=2)
    ax4.set_xlabel('平移误差 (m)')
    ax4.set_ylabel('累积百分比 (%)')
    ax4.set_title('平移误差累积分布 (CDF)')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[可视化] 已保存: {save_path}")
    
    plt.show()
    plt.close()


def visualize_error_scatter(
    rot_errors: np.ndarray,
    trans_errors: np.ndarray,
    rot_thresh: float = 5.0,
    trans_thresh: float = 2.0,
    title: str = "误差散点图",
    save_path: str = None,
):
    """
    旋转-平移误差散点图
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 判断成功/失败
    success = (rot_errors < rot_thresh) & (trans_errors < trans_thresh)
    
    # 绘制散点
    ax.scatter(rot_errors[~success], trans_errors[~success], 
               c='red', s=20, alpha=0.5, label='失败')
    ax.scatter(rot_errors[success], trans_errors[success], 
               c='green', s=20, alpha=0.5, label='成功')
    
    # 阈值线
    ax.axvline(rot_thresh, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.axhline(trans_thresh, color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    # 填充成功区域
    ax.fill_between([0, rot_thresh], 0, trans_thresh, color='green', alpha=0.1)
    
    ax.set_xlabel('旋转误差 (°)', fontsize=12)
    ax.set_ylabel('平移误差 (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 计算并显示召回率
    recall = np.mean(success) * 100
    ax.text(0.95, 0.05, f'Recall: {recall:.2f}%', 
            transform=ax.transAxes, fontsize=12, 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[可视化] 已保存: {save_path}")
    
    plt.show()
    plt.close()


# =============================================================================
# 3. 召回率曲线
# =============================================================================

def visualize_recall_curve(
    recall_data: Dict[str, np.ndarray],
    title: str = "召回率曲线",
    save_path: str = None,
):
    """
    可视化召回率曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 旋转阈值 vs 召回率
    ax1 = axes[0]
    ax1.plot(recall_data['rot_thresholds'], recall_data['rot_recall'], 
             color='steelblue', linewidth=2, marker='o', markersize=4)
    ax1.axvline(5.0, color='red', linestyle='--', alpha=0.7, label='默认阈值 5°')
    ax1.set_xlabel('旋转阈值 (°)')
    ax1.set_ylabel('召回率 (%)')
    ax1.set_title('旋转阈值 vs 召回率 (固定平移阈值=2m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # 平移阈值 vs 召回率
    ax2 = axes[1]
    ax2.plot(recall_data['trans_thresholds'], recall_data['trans_recall'], 
             color='seagreen', linewidth=2, marker='o', markersize=4)
    ax2.axvline(2.0, color='red', linestyle='--', alpha=0.7, label='默认阈值 2m')
    ax2.set_xlabel('平移阈值 (m)')
    ax2.set_ylabel('召回率 (%)')
    ax2.set_title('平移阈值 vs 召回率 (固定旋转阈值=5°)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[可视化] 已保存: {save_path}")
    
    plt.show()
    plt.close()


# =============================================================================
# 4. Attention Map 可视化
# =============================================================================

def visualize_attention_map(
    attention: np.ndarray,
    src_centroids: np.ndarray = None,
    tgt_centroids: np.ndarray = None,
    title: str = "Attention Map",
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    可视化 Attention Map
    
    Args:
        attention: Attention 权重 (N, M) 或 (H, N, M)
        src_centroids: 源点质心 (N, 3)
        tgt_centroids: 目标点质心 (M, 3)
    """
    # 如果是多头，取平均
    if attention.ndim == 3:
        attention = attention.mean(axis=0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(attention, cmap='hot', aspect='auto')
    ax.set_xlabel('Target Patch Index')
    ax.set_ylabel('Source Patch Index')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[可视化] 已保存: {save_path}")
    
    plt.show()
    plt.close()


def visualize_correspondence(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    correspondence_weights: np.ndarray,
    top_k: int = 50,
    title: str = "对应关系可视化",
    save_path: str = None,
):
    """
    可视化点对应关系
    
    Args:
        src_points: 源点云 (N, 3)
        tgt_points: 目标点云 (M, 3)
        correspondence_weights: 对应权重 (N, M)
        top_k: 显示前 k 个最强对应
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云
    ax.scatter(src_points[:, 0], src_points[:, 1], src_points[:, 2],
               c='blue', s=1, alpha=0.3, label='Source')
    ax.scatter(tgt_points[:, 0], tgt_points[:, 1], tgt_points[:, 2],
               c='green', s=1, alpha=0.3, label='Target')
    
    # 获取最强对应
    N = correspondence_weights.shape[0]
    best_matches = np.argmax(correspondence_weights, axis=1)
    max_weights = np.max(correspondence_weights, axis=1)
    
    # 选择 top_k 个最强对应
    top_indices = np.argsort(max_weights)[-top_k:]
    
    # 绘制对应线
    for i in top_indices:
        j = best_matches[i]
        weight = max_weights[i]
        
        # 颜色根据权重
        color = plt.cm.Reds(weight)
        
        ax.plot([src_points[i, 0], tgt_points[j, 0]],
                [src_points[i, 1], tgt_points[j, 1]],
                [src_points[i, 2], tgt_points[j, 2]],
                c=color, linewidth=0.5, alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[可视化] 已保存: {save_path}")
    
    plt.show()
    plt.close()


# =============================================================================
# 5. 训练曲线可视化
# =============================================================================

def visualize_training_curves(
    train_losses: List[float],
    val_losses: List[float] = None,
    rot_losses: List[float] = None,
    trans_losses: List[float] = None,
    learning_rates: List[float] = None,
    title: str = "训练曲线",
    save_path: str = None,
):
    """
    可视化训练曲线
    """
    n_plots = 1
    if rot_losses is not None:
        n_plots += 1
    if learning_rates is not None:
        n_plots += 1
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    
    epochs = range(1, len(train_losses) + 1)
    
    # 总损失
    axes[0].plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    if val_losses is not None:
        axes[0].plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('总损失')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 分项损失
    idx = 1
    if rot_losses is not None and trans_losses is not None:
        axes[idx].plot(epochs, rot_losses, 'b-', linewidth=2, label='Rotation Loss')
        axes[idx].plot(epochs, trans_losses, 'g-', linewidth=2, label='Translation Loss')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Loss')
        axes[idx].set_title('分项损失')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        idx += 1
    
    # 学习率
    if learning_rates is not None:
        axes[idx].plot(epochs, learning_rates, 'purple', linewidth=2)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Learning Rate')
        axes[idx].set_title('学习率')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_yscale('log')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[可视化] 已保存: {save_path}")
    
    plt.show()
    plt.close()


# =============================================================================
# 6. 重叠分数可视化
# =============================================================================

def visualize_overlap_scores(
    points: np.ndarray,
    scores: np.ndarray,
    title: str = "重叠分数可视化",
    save_path: str = None,
):
    """
    可视化点云的重叠分数
    
    Args:
        points: 点云坐标 (N, 3)
        scores: 重叠分数 (N,) 范围 [0, 1]
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 根据分数着色
    colors = plt.cm.RdYlGn(scores)  # 红(低)->黄->绿(高)
    
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c=scores, cmap='RdYlGn', s=5, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Overlap Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[可视化] 已保存: {save_path}")
    
    plt.show()
    plt.close()


# =============================================================================
# 综合可视化类
# =============================================================================

class RegistrationVisualizer:
    """
    点云配准可视化工具类
    设计者：周女士
    """
    
    def __init__(self, save_dir: str = './visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        print(f"[RegistrationVisualizer] 保存目录: {save_dir}")
    
    def visualize_sample(
        self,
        src_points: np.ndarray,
        tgt_points: np.ndarray,
        pred_R: np.ndarray,
        pred_t: np.ndarray,
        gt_R: np.ndarray = None,
        gt_t: np.ndarray = None,
        sample_name: str = 'sample',
    ):
        """可视化单个样本的配准结果"""
        # 变换源点云
        pred_transformed = (pred_R @ src_points.T).T + pred_t.flatten()
        
        # 保存路径
        save_path = os.path.join(self.save_dir, f'{sample_name}_registration.png')
        
        if gt_R is not None:
            gt_transformed = (gt_R @ src_points.T).T + gt_t.flatten()
            visualize_registration_comparison(
                src_points, tgt_points, pred_transformed, gt_transformed,
                title=f"配准对比 - {sample_name}",
                save_path=save_path
            )
        else:
            visualize_point_clouds(
                src_points, tgt_points, pred_transformed,
                title=f"配准结果 - {sample_name}",
                save_path=save_path
            )
    
    def visualize_metrics(
        self,
        rot_errors: np.ndarray,
        trans_errors: np.ndarray,
        rot_thresh: float = 5.0,
        trans_thresh: float = 2.0,
        prefix: str = '',
    ):
        """可视化评估指标"""
        # 误差分布
        visualize_error_distribution(
            rot_errors, trans_errors, rot_thresh, trans_thresh,
            title=f"{prefix}误差分布",
            save_path=os.path.join(self.save_dir, f'{prefix}error_distribution.png')
        )
        
        # 误差散点图
        visualize_error_scatter(
            rot_errors, trans_errors, rot_thresh, trans_thresh,
            title=f"{prefix}误差散点图",
            save_path=os.path.join(self.save_dir, f'{prefix}error_scatter.png')
        )
        
        # 召回率曲线
        from utils.metrics import registration_recall_curve
        recall_data = registration_recall_curve(rot_errors, trans_errors)
        visualize_recall_curve(
            recall_data,
            title=f"{prefix}召回率曲线",
            save_path=os.path.join(self.save_dir, f'{prefix}recall_curve.png')
        )


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("测试可视化工具...")
    
    np.random.seed(42)
    
    # 模拟数据
    N = 1000
    src_points = np.random.randn(N, 3).astype(np.float32)
    
    # 随机变换
    angle = 0.1
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    t = np.array([1.0, 0.5, 0.2])
    
    tgt_points = (R @ src_points.T).T + t
    transformed_src = (R @ src_points.T).T + t + np.random.randn(N, 3) * 0.1
    
    # 测试点云可视化
    print("\n1. 测试点云可视化...")
    visualize_point_clouds(src_points, tgt_points, transformed_src,
                           title="测试 - 点云配准", save_path=None)
    
    # 测试误差分布
    print("\n2. 测试误差分布...")
    rot_errors = np.abs(np.random.randn(100) * 3)
    trans_errors = np.abs(np.random.randn(100) * 1.5)
    visualize_error_distribution(rot_errors, trans_errors, save_path=None)
    
    # 测试误差散点图
    print("\n3. 测试误差散点图...")
    visualize_error_scatter(rot_errors, trans_errors, save_path=None)
    
    print("\n可视化测试完成！✓")