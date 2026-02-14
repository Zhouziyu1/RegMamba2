# -*- coding:UTF-8 -*-
"""
RegMamba 配置文件
设计者：周女士
"""

import argparse


def regmamba_args():
    parser = argparse.ArgumentParser(description='RegMamba Configuration')
    
    # ========== GPU 设置 ==========
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--multi_gpu', type=str, default=None, help='多GPU，如 "0,1"')
    
    # ========== 数据集设置 ==========
    parser.add_argument('--dataset', type=str, default='kitti', 
                        choices=['kitti', 'nuscenes', '3dmatch'])
    parser.add_argument('--lidar_root', type=str, 
                        default='/home/LY/ZiyuZhou/RegFormer-main/Reg_Mamaba/data/kitti/dataset/sequences',)
    parser.add_argument('--data_list', type=str, default='/home/LY/ZiyuZhou/RegFormer-main/Reg_Mamaba/data/kitti_list')
    parser.add_argument('--num_points', type=int, default=14400, 
                        help='输入点数 N')
    parser.add_argument('--voxel_size', type=float, default=0.3)
    parser.add_argument('--augment', type=float, default=0.5, 
                        help='数据增强概率')
    
    # ========== 模型架构 ==========
    parser.add_argument('--patch_size', type=int, default=32, help='Patch大小 P')
    parser.add_argument('--stride', type=int, default=16, help='步长 S (50%重叠)')
    parser.add_argument('--d_model', type=int, default=256, help='特征维度 D')
    parser.add_argument('--n_mamba_layers', type=int, default=4, help='Mamba层数')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # ========== 损失函数权重 ==========
    parser.add_argument('--lambda_rot', type=float, default=1.0, help='旋转损失权重')
    parser.add_argument('--lambda_trans', type=float, default=1.0, help='平移损失权重')
    parser.add_argument('--lambda_overlap', type=float, default=0.5, help='重叠损失权重')
    parser.add_argument('--lambda_ds', type=float, default=0.1, help='深度监督权重')
    parser.add_argument('--use_deep_supervision', type=bool, default=True)
    parser.add_argument('--use_overlap_loss', type=bool, default=False, 
                        help='是否使用重叠监督（需要GT mask）')
    
    # ========== 训练设置 ==========
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    
    # ========== 优化器设置 ==========
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr_stepsize', type=int, default=20)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    
    # ========== 其他设置 ==========
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None, help='恢复训练的检查点')
    parser.add_argument('--eval_interval', type=int, default=2, help='验证间隔')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
    parser.add_argument('--visualize', action='store_true', help='生成可视化结果')
    
    args = parser.parse_args()
    return args