# -*- coding:UTF-8 -*-
"""
RegMamba 训练脚本
================================================================================
设计者：周女士

使用方法:
    python train.py --dataset kitti --batch_size 8 --max_epoch 100
================================================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import time
import datetime
from tqdm import tqdm

# 导入模型和损失函数
from model.regmamba import RegMamba, RegMambaConfig
from model.losses import RegMambaLoss

# 导入配置
from config import regmamba_args

# 导入数据集
from data.kitti_data import KittiDataset
from data.nuscenes_data import NuscenesDataset

# 导入工具
from tools.logger_tools import log_print, creat_logger
from tools.euler_tools import quat2mat
from tools.metrics import RegistrationMetrics, quat2mat
from tools.visualization import (
    visualize_training_curves,
    visualize_error_distribution,
    visualize_error_scatter,
    visualize_point_clouds,
    visualize_overlap_scores,
)
from tools.excel_logger import ExcelLogger
import time

def calc_error_np(pred_R, pred_t, gt_R, gt_t):
    """计算旋转和平移误差"""
    tmp = (np.trace(pred_R.transpose().dot(gt_R)) - 1) / 2
    tmp = np.clip(tmp, -1.0, 1.0)
    L_rot = np.arccos(tmp) * 180 / np.pi
    L_trans = np.linalg.norm(pred_t - gt_t)
    return L_rot, L_trans

def validate(model, val_loader, criterion, logger, args, excel_logger, epoch):
    """验证函数 - 包含完整评价指标并记录到Excel"""
    model.eval()
    
    rot_errors = []
    trans_errors = []
    point_rmse_list = []
    point_mae_list = []
    total_loss = 0.0
    inference_times = []
    n_samples = 0
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc='Validating'):
            src_points = batch_data['src_points'].cuda()
            tgt_points = batch_data['tgt_points'].cuda()
            gt_q = batch_data['gt_quaternion'].cuda()
            gt_t = batch_data['gt_translation'].cuda()
            
            # 计时
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 前向传播
            output = model(src_points, tgt_points)
            
            torch.cuda.synchronize()
            batch_time = time.time() - start_time
            inference_times.append(batch_time / src_points.size(0))
            
            # 损失
            loss_dict = criterion(
                pred_q=output['quaternion'],
                pred_t=output['translation'],
                gt_q=gt_q,
                gt_t=gt_t,
            )
            total_loss += loss_dict['total_loss'].item() * src_points.size(0)
            
            # 计算误差
            pred_q_np = output['quaternion'].cpu().numpy()
            pred_t_np = output['translation'].cpu().numpy()
            gt_q_np = gt_q.cpu().numpy()
            gt_t_np = gt_t.cpu().numpy()
            src_np = src_points.cpu().numpy()
            
            for i in range(src_points.size(0)):
                pred_R = quat2mat(pred_q_np[i])
                gt_R = quat2mat(gt_q_np[i])
                
                # 旋转误差
                R_diff = np.dot(pred_R.T, gt_R)
                trace = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
                rot_err = np.arccos(trace) * 180.0 / np.pi
                rot_errors.append(rot_err)
                
                # 平移误差
                trans_err = np.linalg.norm(pred_t_np[i] - gt_t_np[i])
                trans_errors.append(trans_err)
                
                # 点级RMSE
                pred_transformed = (pred_R @ src_np[i].T).T + pred_t_np[i]
                gt_transformed = (gt_R @ src_np[i].T).T + gt_t_np[i]
                point_diff = np.linalg.norm(pred_transformed - gt_transformed, axis=1)
                point_rmse_list.append(np.sqrt(np.mean(point_diff ** 2)))
                point_mae_list.append(np.mean(point_diff))
            
            n_samples += src_points.size(0)
    
    # 转为numpy
    rot_errors = np.array(rot_errors)
    trans_errors = np.array(trans_errors)
    
    # 计算召回率
    recall = np.mean((rot_errors < args.rot_thresh) & (trans_errors < args.trans_thresh)) * 100
    
    # 平均推理时间
    avg_inference_time = np.mean(inference_times) * 1000  # ms
    
    # ===== 记录到 Excel =====
    excel_logger.log_val_epoch(
        epoch=epoch,
        val_loss=total_loss / n_samples,
        rot_errors=rot_errors,
        trans_errors=trans_errors,
        rot_thresh=args.rot_thresh,
        trans_thresh=args.trans_thresh,
        point_rmse=np.mean(point_rmse_list),
        point_mae=np.mean(point_mae_list),
        inference_time=avg_inference_time,
    )
    
    # 打印
    logger.info(f'  验证: Loss={total_loss/n_samples:.4f}, Recall={recall:.2f}%, '
                f'RRE={np.mean(rot_errors):.4f}°, RTE={np.mean(trans_errors):.4f}m')
    
    return {
        'loss': total_loss / n_samples,
        'recall': recall,
        'rot_error': np.mean(rot_errors),
        'trans_error': np.mean(trans_errors),
    }



def main():
    """主训练函数"""
    # 解析参数
    best_recall = 0.0
    args = regmamba_args()
    print(f"[CHECK] args.dataset = {args.dataset}")
    # ========== 创建实验目录 ==========
    base_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = os.path.join(base_dir, 'experiment')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # ========== Excel 日志记录器 ==========
    excel_logger = ExcelLogger(
        save_dir='./experiment/logs',
        exp_name='RegMamba_KITTI',
        auto_save=True,
    )

    if not args.task_name:
        task_name = f'RegMamba_{args.dataset}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'
    else:
        task_name = args.task_name
    
    file_dir = os.path.join(experiment_dir, task_name)
    os.makedirs(file_dir, exist_ok=True)
    
    log_dir = os.path.join(file_dir, 'logs')
    checkpoints_dir = os.path.join(file_dir, 'checkpoints')
    eval_dir = os.path.join(file_dir, 'eval')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # 创建日志
    logger = creat_logger(log_dir, 'RegMamba')
    logger.info('=' * 70)
    logger.info('  RegMamba 训练开始')
    logger.info('  设计者：周女士')
    logger.info('=' * 70)
    logger.info(f'\n配置参数:\n{args}\n')

        # ========== 训练记录 ==========
    history = {
        'train_loss': [],
        'val_loss': [],
        'rot_loss': [],
        'trans_loss': [],
        'learning_rate': [],
        'val_rot_error': [],
        'val_trans_error': [],
        'val_recall': [],
    }
    
    # ========== 数据集 ==========
    if args.dataset == 'kitti':
       # print(f"✅ DEBUG: data_list = {args.data_list} (type: {type(args.data_list)})")
        train_dataset = KittiDataset(
            args.lidar_root, 'train', args.num_points,
            args.voxel_size,args.data_list, args.augment
        )
        val_dataset = KittiDataset(
            args.lidar_root, 'val', args.num_points,
            args.voxel_size,args.data_list, augment=0.0
        )
    elif args.dataset == 'nuscenes':
        train_dataset = NuscenesDataset(
            args.lidar_root, 'train', args.num_points,
            args.voxel_size, args.data_list, args.augment
        )
        val_dataset = NuscenesDataset(
            args.lidar_root, 'val', args.num_points,
            args.voxel_size, args.data_list, augment=0.0
        )
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    # print(f"[DEBUG] args.num_points = {args.num_points}")

    sample = train_dataset[0]
    print(f"[CHECK] train_dataset[0] type: {type(sample)}")
    if isinstance(sample, dict):
        print(f"[CHECK] sample keys: {sample.keys()}")
    else:
        print(f"[CHECK] sample length: {len(sample)}")

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,   # 不要硬编码 1
    shuffle=True,#是否打乱数据
    num_workers=args.workers,
    pin_memory=True,
    drop_last=True # 推荐，保证 batch 尺寸一致
    )    
    val_loader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=args.batch_size, 
    num_workers=args.workers, 
    shuffle=False, 
    pin_memory=True,
    drop_last=True)
    
    logger.info(f'训练集大小: {len(train_dataset)}')
    logger.info(f'验证集大小: {len(val_dataset)}')
    
    # ========== 模型 ==========
    config = RegMambaConfig(
        n_points=args.num_points,
        patch_size=args.patch_size,
        stride=args.stride,
        d_model=args.d_model,
        n_mamba_layers=args.n_mamba_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    
    model = RegMamba(config)
    
    # GPU 设置
    torch.cuda.set_device(args.gpu)
    model = model.cuda()
    logger.info(f'使用单GPU: {args.gpu}')
    
    # ========== 损失函数 ==========
    criterion = RegMambaLoss(
        lambda_rot=args.lambda_rot,
        lambda_trans=args.lambda_trans,
        lambda_overlap=args.lambda_overlap,
        lambda_ds=args.lambda_ds,
        use_deep_supervision=args.use_deep_supervision,
        use_overlap_loss=args.use_overlap_loss,
    )
    
    # ========== 优化器 ==========
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_stepsize,
        gamma=args.lr_gamma,
    )
    
    # ========== 加载检查点 ==========
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.ckpt:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        if 'history' in checkpoint:
            history = checkpoint['history']
            logger.info(f'  恢复训练历史记录，共 {len(history["train_loss"])} 个 epoch')
        logger.info(f'从检查点恢复: {args.ckpt}, epoch={start_epoch}')
    
    # ========== 训练循环 ==========
    logger.info('\n开始训练...\n')
    
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        epoch_start_time = time.time()
        
        # 损失累计
        total_loss = 0.0
        total_rot_loss = 0.0
        total_trans_loss = 0.0
        total_overlap_loss = 0.0
        total_ds_loss = 0.0
        feature_losses_sum = [0.0, 0.0, 0.0, 0.0]  # 4层Mamba
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.max_epoch}')
        
        for batch_data in pbar:
            src_points = batch_data['src_points'].cuda()
            tgt_points = batch_data['tgt_points'].cuda()
            gt_q = batch_data['gt_quaternion'].cuda()
            gt_t = batch_data['gt_translation'].cuda()
            
            # 前向传播
            output = model(src_points, tgt_points)
            
            # 计算损失
            loss_dict = criterion(
                pred_q=output['quaternion'],
                pred_t=output['translation'],
                gt_q=gt_q,
                gt_t=gt_t,
                overlap_scores=output.get('overlap_scores'),
                src_intermediate=output.get('src_intermediate_feats'),
                tgt_intermediate=output.get('tgt_intermediate_feats'),
            )
            
            loss = loss_dict['total_loss']
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            total_rot_loss += loss_dict['rot_loss'].item()
            total_trans_loss += loss_dict['trans_loss'].item()
            total_overlap_loss += loss_dict.get('overlap_loss', torch.tensor(0.0)).item()
            total_ds_loss += loss_dict.get('ds_loss', torch.tensor(0.0)).item()
            
            # 如果有逐层特征损失
            if 'feature_losses' in loss_dict:
                for i, fl in enumerate(loss_dict['feature_losses'][:4]):
                    feature_losses_sum[i] += fl.item()
            
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rot': f'{loss_dict["rot_loss"].item():.4f}',
                'trans': f'{loss_dict["trans_loss"].item():.4f}',
            })
        
        # ===== Epoch 统计 =====
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / n_batches
        avg_rot_loss = total_rot_loss / n_batches
        avg_trans_loss = total_trans_loss / n_batches
        avg_overlap_loss = total_overlap_loss / n_batches
        avg_ds_loss = total_ds_loss / n_batches
        avg_feature_losses = [fl / n_batches for fl in feature_losses_sum]
        current_lr = optimizer.param_groups[0]['lr']
        
        # ===== 记录到 Excel =====
        excel_logger.log_train_epoch(
            epoch=epoch + 1,
            learning_rate=current_lr,
            time_elapsed=epoch_time,
            total_loss=avg_loss,
            rot_loss=avg_rot_loss,
            trans_loss=avg_trans_loss,
            overlap_loss=avg_overlap_loss,
            ds_loss=avg_ds_loss,
            feature_losses=avg_feature_losses,
        )
        
        logger.info(f'Epoch {epoch+1}: loss={avg_loss:.4f}, rot={avg_rot_loss:.4f}, '
                    f'trans={avg_trans_loss:.4f}, time={epoch_time:.1f}s')
        
        # 学习率更新
        scheduler.step()
        
        # ===== 验证 =====
        if (epoch + 1) % args.eval_interval == 0:
            val_results = validate(model, val_loader, criterion, logger, args, excel_logger, epoch + 1)
            
            # 保存最优模型
            if val_results['recall'] > best_recall:
                best_recall = val_results['recall']
                save_path = os.path.join(checkpoints_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict() if args.multi_gpu else model.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_recall': best_recall,
                }, save_path)
                logger.info(f'  ★ 保存最优模型 (Recall={best_recall:.2f}%): {save_path}')
        
        # 定期保存
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(checkpoints_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if args.multi_gpu else model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, save_path)

            # ===== 可视化训练曲线 =====
            visualize_training_curves(
                train_losses=history['train_loss'],
                val_losses=history['val_loss'] if history['val_loss'] else None,
                rot_losses=history['rot_loss'],
                trans_losses=history['trans_loss'],
                learning_rates=history['learning_rate'],
                title=f'训练曲线 - Epoch {epoch+1}',
                save_path=os.path.join(eval_dir, f'training_curves_epoch{epoch+1}.png'),
            )
            logger.info(f'  保存检查点: {save_path}')
    
    # ===== 训练结束 =====
    logger.info(f'\n训练完成！最优召回率: {best_recall:.2f}%')
    logger.info(f'Excel日志: {excel_logger.excel_path}')

    # 可视化训练历史
    import matplotlib.pyplot as plt
    visualize_training_curves(
        train_losses=history['train_loss'],
        val_losses=history['val_loss'] if history['val_loss'] else None,
        rot_losses=history['rot_loss'],
        trans_losses=history['trans_loss'],
        learning_rates=history['learning_rate'],
        title='训练曲线',
        save_path=os.path.join(eval_dir, 'training_curves.png'),
    )

    logger.info('\n训练完成！')
    return avg_loss


if __name__ == '__main__':
    main()