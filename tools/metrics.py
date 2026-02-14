# -*- coding:UTF-8 -*-
"""
点云配准评价指标
================================================================================
设计者：周女士

包含以下评价指标：
1. 旋转误差 (Rotation Error) - 角度制
2. 平移误差 (Translation Error) - 米
3. 配准召回率 (Registration Recall) - RR
4. 相对旋转误差 (Relative Rotation Error) - RRE
5. 相对平移误差 (Relative Translation Error) - RTE
6. 各向同性变换误差 (Isotropic Transformation Error) - ITE
7. 均方根误差 (RMSE)
8. Chamfer Distance (倒角距离)
9. 成功率曲线 (Success Rate Curve)
================================================================================
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree


# =============================================================================
# 基础旋转/平移误差
# =============================================================================

def rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    计算旋转误差 (角度制)
    
    公式: θ = arccos((trace(R_pred^T @ R_gt) - 1) / 2)
    
    Args:
        R_pred: 预测旋转矩阵 (3, 3)
        R_gt: 真实旋转矩阵 (3, 3)
    
    Returns:
        error: 旋转误差 (度)
    """
    R_diff = np.dot(R_pred.T, R_gt)
    trace = np.trace(R_diff)
    trace = np.clip((trace - 1) / 2, -1.0, 1.0)
    error_rad = np.arccos(trace)
    error_deg = error_rad * 180.0 / np.pi
    return error_deg


def translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """
    计算平移误差 (欧氏距离)
    
    Args:
        t_pred: 预测平移向量 (3,)
        t_gt: 真实平移向量 (3,)
    
    Returns:
        error: 平移误差 (米)
    """
    return np.linalg.norm(t_pred.flatten() - t_gt.flatten())


def relative_rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    相对旋转误差 (RRE) - 同 rotation_error
    常用于 3DMatch/KITTI 评估
    """
    return rotation_error(R_pred, R_gt)


def relative_translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """
    相对平移误差 (RTE) - 同 translation_error
    常用于 3DMatch/KITTI 评估
    """
    return translation_error(t_pred, t_gt)


# =============================================================================
# 配准召回率 (Registration Recall)
# =============================================================================

def registration_recall(
    rot_errors: np.ndarray,
    trans_errors: np.ndarray,
    rot_thresh: float = 5.0,      # 度
    trans_thresh: float = 2.0,    # 米 (KITTI) 或 0.3m (3DMatch)
) -> Tuple[float, np.ndarray]:
    """
    计算配准召回率 (Registration Recall)
    
    定义: 成功配准的样本比例
    成功条件: 旋转误差 < rot_thresh AND 平移误差 < trans_thresh
    
    Args:
        rot_errors: 所有样本的旋转误差数组 (N,)
        trans_errors: 所有样本的平移误差数组 (N,)
        rot_thresh: 旋转阈值 (度)
        trans_thresh: 平移阈值 (米)
    
    Returns:
        recall: 召回率 (0~100%)
        success_mask: 成功样本的布尔掩码
    """
    success_mask = (rot_errors < rot_thresh) & (trans_errors < trans_thresh)
    recall = np.mean(success_mask) * 100.0
    return recall, success_mask


def registration_recall_curve(
    rot_errors: np.ndarray,
    trans_errors: np.ndarray,
    rot_thresholds: np.ndarray = None,
    trans_thresholds: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """
    计算不同阈值下的召回率曲线
    
    Args:
        rot_errors: 旋转误差数组
        trans_errors: 平移误差数组
        rot_thresholds: 旋转阈值数组
        trans_thresholds: 平移阈值数组
    
    Returns:
        curves: {
            'rot_thresholds': [...],
            'trans_thresholds': [...],
            'rot_recall': [...],  # 固定 trans_thresh 变化 rot_thresh
            'trans_recall': [...],  # 固定 rot_thresh 变化 trans_thresh
        }
    """
    if rot_thresholds is None:
        rot_thresholds = np.arange(0.5, 15.5, 0.5)  # 0.5° ~ 15°
    if trans_thresholds is None:
        trans_thresholds = np.arange(0.1, 3.1, 0.1)  # 0.1m ~ 3m
    
    # 固定平移阈值=2m，变化旋转阈值
    rot_recall = []
    for rot_th in rot_thresholds:
        recall, _ = registration_recall(rot_errors, trans_errors, rot_th, 2.0)
        rot_recall.append(recall)
    
    # 固定旋转阈值=5°，变化平移阈值
    trans_recall = []
    for trans_th in trans_thresholds:
        recall, _ = registration_recall(rot_errors, trans_errors, 5.0, trans_th)
        trans_recall.append(recall)
    
    return {
        'rot_thresholds': rot_thresholds,
        'trans_thresholds': trans_thresholds,
        'rot_recall': np.array(rot_recall),
        'trans_recall': np.array(trans_recall),
    }


# =============================================================================
# 点云级别的误差
# =============================================================================

def transformed_point_error(
    src_points: np.ndarray,      # (N, 3)
    T_pred: np.ndarray,          # (4, 4) 预测变换矩阵
    T_gt: np.ndarray,            # (4, 4) 真实变换矩阵
) -> Dict[str, float]:
    """
    计算变换后点云的误差
    
    Args:
        src_points: 源点云 (N, 3)
        T_pred: 预测的 4x4 变换矩阵
        T_gt: 真实的 4x4 变换矩阵
    
    Returns:
        errors: {
            'rmse': 均方根误差,
            'mae': 平均绝对误差,
            'max': 最大误差,
        }
    """
    N = src_points.shape[0]
    
    # 转为齐次坐标
    src_homo = np.hstack([src_points, np.ones((N, 1))])  # (N, 4)
    
    # 变换
    pred_transformed = (T_pred @ src_homo.T).T[:, :3]  # (N, 3)
    gt_transformed = (T_gt @ src_homo.T).T[:, :3]       # (N, 3)
    
    # 计算误差
    diff = pred_transformed - gt_transformed
    distances = np.linalg.norm(diff, axis=1)
    
    return {
        'rmse': np.sqrt(np.mean(distances ** 2)),
        'mae': np.mean(distances),
        'max': np.max(distances),
        'median': np.median(distances),
    }


def chamfer_distance(
    pc1: np.ndarray,  # (N, 3)
    pc2: np.ndarray,  # (M, 3)
) -> Dict[str, float]:
    """
    计算 Chamfer Distance (倒角距离)
    
    CD = (1/N) * Σ min_j ||p1_i - p2_j||^2 + (1/M) * Σ min_i ||p2_j - p1_i||^2
    
    Args:
        pc1: 点云1 (N, 3)
        pc2: 点云2 (M, 3)
    
    Returns:
        distances: {
            'chamfer': 双向 Chamfer Distance,
            'chamfer_pc1_to_pc2': pc1 到 pc2 的单向距离,
            'chamfer_pc2_to_pc1': pc2 到 pc1 的单向距离,
        }
    """
    # 构建 KD-Tree
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    # pc1 到 pc2 的最近邻距离
    dist1, _ = tree2.query(pc1, k=1)
    chamfer_1_to_2 = np.mean(dist1 ** 2)
    
    # pc2 到 pc1 的最近邻距离
    dist2, _ = tree1.query(pc2, k=1)
    chamfer_2_to_1 = np.mean(dist2 ** 2)
    
    return {
        'chamfer': chamfer_1_to_2 + chamfer_2_to_1,
        'chamfer_pc1_to_pc2': chamfer_1_to_2,
        'chamfer_pc2_to_pc1': chamfer_2_to_1,
    }


# =============================================================================
# 综合评估类
# =============================================================================

class RegistrationMetrics:
    """
    点云配准综合评估类
    设计者：周女士
    
    使用方法:
        metrics = RegistrationMetrics()
        metrics.add_sample(R_pred, t_pred, R_gt, t_gt, src_points)
        results = metrics.compute()
        metrics.print_summary()
    """
    
    def __init__(
        self,
        rot_thresh: float = 5.0,
        trans_thresh: float = 2.0,
        dataset_name: str = 'KITTI',
    ):
        self.rot_thresh = rot_thresh
        self.trans_thresh = trans_thresh
        self.dataset_name = dataset_name
        
        # 存储所有样本的误差
        self.rot_errors = []
        self.trans_errors = []
        self.rmse_errors = []
        self.mae_errors = []
        self.inference_times = []
        
        # 原始预测和真值 (用于详细分析)
        self.predictions = []
        self.ground_truths = []
        
    def add_sample(
        self,
        R_pred: np.ndarray,
        t_pred: np.ndarray,
        R_gt: np.ndarray,
        t_gt: np.ndarray,
        src_points: np.ndarray = None,
        inference_time: float = None,
    ):
        """添加一个样本的预测结果"""
        # 计算旋转和平移误差
        rot_err = rotation_error(R_pred, R_gt)
        trans_err = translation_error(t_pred, t_gt)
        
        self.rot_errors.append(rot_err)
        self.trans_errors.append(trans_err)
        
        # 如果提供了源点云，计算点级误差
        if src_points is not None:
            T_pred = np.eye(4)
            T_pred[:3, :3] = R_pred
            T_pred[:3, 3] = t_pred.flatten()
            
            T_gt = np.eye(4)
            T_gt[:3, :3] = R_gt
            T_gt[:3, 3] = t_gt.flatten()
            
            point_errors = transformed_point_error(src_points, T_pred, T_gt)
            self.rmse_errors.append(point_errors['rmse'])
            self.mae_errors.append(point_errors['mae'])
        
        # 推理时间
        if inference_time is not None:
            self.inference_times.append(inference_time)
        
        # 保存原始数据
        self.predictions.append({'R': R_pred.copy(), 't': t_pred.copy()})
        self.ground_truths.append({'R': R_gt.copy(), 't': t_gt.copy()})
    
    def compute(self) -> Dict[str, float]:
        """计算所有指标"""
        rot_errors = np.array(self.rot_errors)
        trans_errors = np.array(self.trans_errors)
        
        # 配准召回率
        recall, success_mask = registration_recall(
            rot_errors, trans_errors,
            self.rot_thresh, self.trans_thresh
        )
        
        success_indices = np.where(success_mask)[0]
        
        results = {
            # 总体指标
            'num_samples': len(rot_errors),
            'registration_recall': recall,
            
            # 全部样本统计
            'rot_error_mean': np.mean(rot_errors),
            'rot_error_std': np.std(rot_errors),
            'rot_error_median': np.median(rot_errors),
            'trans_error_mean': np.mean(trans_errors),
            'trans_error_std': np.std(trans_errors),
            'trans_error_median': np.median(trans_errors),
        }
        
        # 成功样本统计
        if len(success_indices) > 0:
            results.update({
                'num_success': len(success_indices),
                'rot_error_mean_success': np.mean(rot_errors[success_indices]),
                'rot_error_std_success': np.std(rot_errors[success_indices]),
                'trans_error_mean_success': np.mean(trans_errors[success_indices]),
                'trans_error_std_success': np.std(trans_errors[success_indices]),
            })
        
        # 点级误差 (如果有)
        if len(self.rmse_errors) > 0:
            results['rmse_mean'] = np.mean(self.rmse_errors)
            results['mae_mean'] = np.mean(self.mae_errors)
        
        # 推理时间 (如果有)
        if len(self.inference_times) > 0:
            results['inference_time_mean_ms'] = np.mean(self.inference_times) * 1000
            results['fps'] = 1.0 / np.mean(self.inference_times)
        
        return results
    
    def print_summary(self):
        """打印评估结果摘要"""
        results = self.compute()
        
        print("\n" + "=" * 70)
        print(f"  配准评估结果 - {self.dataset_name}")
        print(f"  设计者：周女士")
        print("=" * 70)
        
        print(f"\n【基本信息】")
        print(f"  样本数: {results['num_samples']}")
        print(f"  阈值: 旋转 < {self.rot_thresh}°, 平移 < {self.trans_thresh}m")
        
        print(f"\n【配准召回率】")
        print(f"  Registration Recall: {results['registration_recall']:.2f}%")
        
        print(f"\n【全部样本误差】")
        print(f"  旋转误差 (RRE):")
        print(f"    均值: {results['rot_error_mean']:.4f}°")
        print(f"    标准差: {results['rot_error_std']:.4f}°")
        print(f"    中位数: {results['rot_error_median']:.4f}°")
        print(f"  平��误差 (RTE):")
        print(f"    均值: {results['trans_error_mean']:.4f}m")
        print(f"    标准差: {results['trans_error_std']:.4f}m")
        print(f"    中位数: {results['trans_error_median']:.4f}m")
        
        if 'num_success' in results:
            print(f"\n【成功样本误差】(共 {results['num_success']} 个)")
            print(f"  旋转误差: {results['rot_error_mean_success']:.4f}° ± {results['rot_error_std_success']:.4f}°")
            print(f"  平移误差: {results['trans_error_mean_success']:.4f}m ± {results['trans_error_std_success']:.4f}m")
        
        if 'rmse_mean' in results:
            print(f"\n【点级误差】")
            print(f"  RMSE: {results['rmse_mean']:.4f}m")
            print(f"  MAE: {results['mae_mean']:.4f}m")
        
        if 'inference_time_mean_ms' in results:
            print(f"\n【推理速度】")
            print(f"  平均时间: {results['inference_time_mean_ms']:.2f} ms/sample")
            print(f"  FPS: {results['fps']:.2f}")
        
        print("\n" + "=" * 70)
        
        return results
    
    def get_recall_curve(self) -> Dict[str, np.ndarray]:
        """获取召回率曲线数据"""
        rot_errors = np.array(self.rot_errors)
        trans_errors = np.array(self.trans_errors)
        return registration_recall_curve(rot_errors, trans_errors)
    
    def reset(self):
        """重置所有数据"""
        self.rot_errors = []
        self.trans_errors = []
        self.rmse_errors = []
        self.mae_errors = []
        self.inference_times = []
        self.predictions = []
        self.ground_truths = []


# =============================================================================
# 四元数/旋转矩阵转换工具
# =============================================================================

def quat2mat(q: np.ndarray) -> np.ndarray:
    """四元数 [w,x,y,z] 转旋转矩阵"""
    q = q / np.linalg.norm(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float32)


def mat2quat(R: np.ndarray) -> np.ndarray:
    """旋转矩阵转四元数 [w,x,y,z]"""
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("测试评价指标...")
    
    np.random.seed(42)
    
    # 模��数据
    metrics = RegistrationMetrics(rot_thresh=5.0, trans_thresh=2.0, dataset_name='KITTI')
    
    for i in range(100):
        # 随机生成真实位姿
        angle = np.random.uniform(0, 0.1)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        # Rodrigues 公式
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R_gt = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        t_gt = np.random.randn(3) * 5
        
        # 添加噪声作为预测
        R_pred = R_gt + np.random.randn(3, 3) * 0.01
        U, _, Vt = np.linalg.svd(R_pred)
        R_pred = U @ Vt  # 确保正交
        
        t_pred = t_gt + np.random.randn(3) * 0.5
        
        # 源点云
        src_points = np.random.randn(1000, 3).astype(np.float32)
        
        metrics.add_sample(R_pred, t_pred, R_gt, t_gt, src_points, inference_time=0.05)
    
    # 打印结果
    metrics.print_summary()
    
    print("\n测试完成！✓")