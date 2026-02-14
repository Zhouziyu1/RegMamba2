# -*- coding:UTF-8 -*-
"""
RegMamba 测试脚本
================================================================================
设计者：周女士

使用方法:
    python test.py --ckpt experiment/xxx/checkpoints/best_model.pth --dataset kitti
================================================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import time
from tqdm import tqdm
import argparse

# 导入模型
from model.regmamba import RegMamba, RegMambaConfig

# 导入数据集
from data.kitti_data import KittiDataset

# 导入工具
from tools.euler_tools import quat2mat, mat2quat


def parse_args():
    """解析测试参数"""
    parser = argparse.ArgumentParser(description='RegMamba 测试脚本 - 设计者：周女士')
    
    # 必需参数
    parser.add_argument('--ckpt', type=str, required=True, help='模型检查点路径')
    
    # 数据集设置
    parser.add_argument('--dataset', type=str, default='kitti', choices=['kitti', 'nuscenes'])
    parser.add_argument('--lidar_root', type=str, 
                        default='/home/LY/ZiyuZhou/RegFormer-main/kitti/dataset/sequences')
    parser.add_argument('--pose_file', type=str,
                        default='/home/LY/ZiyuZhou/RegFormer-main/kitti/dataset/poses/test_pairs.txt',
                        help='测试集点云对文件路径')
    parser.add_argument('--num_points', type=int, default=14400)
    parser.add_argument('--voxel_size', type=float, default=0.3)
    
    # 模型设置
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_mamba_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)
    
    # 测试设置
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_results', type=bool, default=True)
    parser.add_argument('--output_dir', type=str, default='./test_results')
    
    # 评估阈值
    parser.add_argument('--rot_thresh', type=float, default=5.0, help='旋转成功阈值 (度)')
    parser.add_argument('--trans_thresh', type=float, default=2.0, help='平移成功阈值 (米)')
    
    args = parser.parse_args()
    return args


def calc_error_np(pred_R, pred_t, gt_R, gt_t):
    """
    计算旋转误差和平移误差
    
    Args:
        pred_R: 预测旋转矩阵 (3x3)
        pred_t: 预测平移向量 (3,) 或 (3,1)
        gt_R: 真实旋转矩阵 (3x3)
        gt_t: 真实平移向量 (3,) 或 (3,1)
    
    Returns:
        rot_error: 旋转误差 (度)
        trans_error: 平移误差 (米)
    """
    # 旋转误差
    R_diff = np.dot(pred_R.T, gt_R)
    trace = np.trace(R_diff)
    trace = np.clip((trace - 1) / 2, -1.0, 1.0)
    rot_error = np.arccos(trace) * 180.0 / np.pi
    
    # 平移误差
    pred_t = pred_t.flatten()
    gt_t = gt_t.flatten()
    trans_error = np.linalg.norm(pred_t - gt_t)
    
    return rot_error, trans_error


def pose_matrix_to_quat_trans(pose_matrix):
    """
    将 3x4 或 4x4 位姿矩阵���换为四元数和平移向量
    
    Args:
        pose_matrix: 位姿矩阵 (3x4) 或 (4x4)
    
    Returns:
        quaternion: 四元数 (4,)
        translation: 平移向量 (3,)
    """
    if pose_matrix.shape[0] == 4:
        R = pose_matrix[:3, :3]
        t = pose_matrix[:3, 3]
    else:
        R = pose_matrix[:, :3]
        t = pose_matrix[:, 3]
    
    quaternion = mat2quat(R)
    translation = t
    
    return quaternion, translation


@torch.no_grad()
def test(args):
    """测试主函数"""
    print("=" * 80)
    print("  RegMamba 测试")
    print("  设计者：周女士")
    print("=" * 80)
    
    # ========== 设置设备 ==========
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    print(f"\n使用 GPU: {args.gpu}")
    
    # ========== 创建模型 ==========
    print("\n加载模型...")
    config = RegMambaConfig(
        n_points=args.num_points,
        patch_size=args.patch_size,
        stride=args.stride,
        d_model=args.d_model,
        n_mamba_layers=args.n_mamba_layers,
        n_heads=args.n_heads,
    )
    
    model = RegMamba(config)
    
    # 加载权重
    checkpoint = torch.load(args.ckpt, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"模型加载成功: {args.ckpt}")
    
    # ========== 加载数据集 ==========
    print("\n加载测试数据集...")
    test_dataset = KittiDataset(
        lidar_root=args.lidar_root,
        pose_file=args.pose_file,
        split='test',
        num_points=args.num_points,
        voxel_size=args.voxel_size,
        augment=0.0,  # 测试时不做数据增强
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # ========== 开始测试 ==========
    print("\n开始测试...")
    
    rot_errors = []
    trans_errors = []
    success_list = []
    inference_times = []
    
    all_results = []
    
    for batch_idx, batch_data in enumerate(tqdm(test_loader, desc='Testing')):
        # 数据移到 GPU
        src_points = batch_data['src_points'].to(device)      # [B, N, 3]
        tgt_points = batch_data['tgt_points'].to(device)      # [B, N, 3]
        gt_q = batch_data['gt_quaternion'].numpy()            # [B, 4]
        gt_t = batch_data['gt_translation'].numpy()           # [B, 3]
        
        # 计时
        torch.cuda.synchronize()
        start_time = time.time()
        
        # 前向传播
        output = model(src_points, tgt_points)
        
        torch.cuda.synchronize()
        inference_time = time.time() - start_time
        inference_times.append(inference_time / src_points.size(0))
        
        # 获取预测结果
        pred_q = output['quaternion'].cpu().numpy()           # [B, 4]
        pred_t = output['translation'].cpu().numpy()          # [B, 3]
        
        # 计算每个样本的误差
        for i in range(src_points.size(0)):
            # 四元数转旋转矩阵
            pred_R = quat2mat(pred_q[i])
            gt_R = quat2mat(gt_q[i])
            
            # 计算误差
            rot_err, trans_err = calc_error_np(
                pred_R, pred_t[i],
                gt_R, gt_t[i]
            )
            
            rot_errors.append(rot_err)
            trans_errors.append(trans_err)
            
            # 判断是否成功
            is_success = (rot_err < args.rot_thresh) and (trans_err < args.trans_thresh)
            success_list.append(is_success)
            
            # 保存结果
            if args.save_results:
                result = {
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'pred_quaternion': pred_q[i].tolist(),
                    'pred_translation': pred_t[i].tolist(),
                    'gt_quaternion': gt_q[i].tolist(),
                    'gt_translation': gt_t[i].tolist(),
                    'rot_error': rot_err,
                    'trans_error': trans_err,
                    'success': is_success,
                }
                all_results.append(result)
    
    # ========== 统计结果 ==========
    rot_errors = np.array(rot_errors)
    trans_errors = np.array(trans_errors)
    success_list = np.array(success_list)
    
    # 成功率
    success_rate = np.mean(success_list) * 100
    
    # 成功样本的误差统计
    success_indices = np.where(success_list)[0]
    if len(success_indices) > 0:
        rot_mean = np.mean(rot_errors[success_indices])
        rot_std = np.std(rot_errors[success_indices])
        trans_mean = np.mean(trans_errors[success_indices])
        trans_std = np.std(trans_errors[success_indices])
    else:
        rot_mean = rot_std = trans_mean = trans_std = float('nan')
    
    # 全部样本的误差统计
    rot_mean_all = np.mean(rot_errors)
    rot_median_all = np.median(rot_errors)
    trans_mean_all = np.mean(trans_errors)
    trans_median_all = np.median(trans_errors)
    
    # 平均推理时间
    avg_inference_time = np.mean(inference_times) * 1000  # ms
    
    # ========== 打印结果 ==========
    print("\n" + "=" * 80)
    print("  测试结果")
    print("=" * 80)
    
    print(f"\n【配准成功率】")
    print(f"  Registration Recall: {success_rate:.2f}%")
    print(f"  (阈值: rot < {args.rot_thresh}°, trans < {args.trans_thresh}m)")
    
    print(f"\n【成功样本误差】")
    print(f"  旋转误差 - 均值: {rot_mean:.4f}°, 标准差: {rot_std:.4f}°")
    print(f"  平移误差 - 均值: {trans_mean:.4f}m, 标准差: {trans_std:.4f}m")
    
    print(f"\n【全部样本误差】")
    print(f"  旋转误差 - 均值: {rot_mean_all:.4f}°, 中位数: {rot_median_all:.4f}°")
    print(f"  平移误差 - 均值: {trans_mean_all:.4f}m, 中位数: {trans_median_all:.4f}m")
    
    print(f"\n【推理速度】")
    print(f"  平均推理时间: {avg_inference_time:.2f} ms/sample")
    print(f"  FPS: {1000 / avg_inference_time:.2f}")
    
    # ========== 保存结果 ==========
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 保存统计摘要
        summary_path = os.path.join(args.output_dir, 'test_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("RegMamba 测试结果\n")
            f.write("设计者：周女士\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"检查点: {args.ckpt}\n")
            f.write(f"测试集: {args.pose_file}\n")
            f.write(f"样本数: {len(test_dataset)}\n\n")
            f.write(f"配准成功率: {success_rate:.2f}%\n")
            f.write(f"阈值: rot < {args.rot_thresh}°, trans < {args.trans_thresh}m\n\n")
            f.write(f"成功样本误差:\n")
            f.write(f"  旋转 - 均值: {rot_mean:.4f}°, 标准差: {rot_std:.4f}°\n")
            f.write(f"  平移 - 均值: {trans_mean:.4f}m, 标准差: {trans_std:.4f}m\n\n")
            f.write(f"全部样本误差:\n")
            f.write(f"  旋转 - 均值: {rot_mean_all:.4f}°, 中位数: {rot_median_all:.4f}°\n")
            f.write(f"  平移 - 均值: {trans_mean_all:.4f}m, 中位数: {trans_median_all:.4f}m\n\n")
            f.write(f"推理时间: {avg_inference_time:.2f} ms/sample\n")
        
        print(f"\n结果已保存到: {summary_path}")
        
        # 保存详细结果
        import json
        details_path = os.path.join(args.output_dir, 'test_details.json')
        with open(details_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"详细结果已保存到: {details_path}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    return {
        'success_rate': success_rate,
        'rot_mean': rot_mean,
        'rot_std': rot_std,
        'trans_mean': trans_mean,
        'trans_std': trans_std,
        'avg_inference_time': avg_inference_time,
    }


if __name__ == '__main__':
    args = parse_args()
    results = test(args)