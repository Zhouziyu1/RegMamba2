# -*- coding:UTF-8 -*-
"""
RegMamba 训练脚本
================================================================================
设计者：周女士

使用方法:
    python train.py  --ckpt /root/autodl-fs/Ragmamba/experiment/RegMamba_kitti_2026-03-01_16-18/checkpoints/best_model.pth --dataset kitti --batch_size 4 --max_epoch 100
    python train.py     --dataset kitti     --lidar_root '/home/LY/ZiyuZhou/RegFormer-main/Reg_Mamaba/data/kitti/dataset/sequences'     --data_list '/home/LY/ZiyuZhou/RegFormer-main/Reg_Mamaba/data/kitti_list'     --num_points 10240     --voxel_size 0.3     --patch_size 32     --stride 16     --d_model 128     --n_mamba_layers 6     --n_heads 8     --dropout 0.1     --batch_size 4     --max_epoch 100     --learning_rate 0.0001     --optimizer Adam     --weight_decay 0.0001     --lr_stepsize 20     --lr_gamma 0.5     --lambda_rot 1.0     --lambda_trans 1.0     --lambda_overlap 0.5     --lambda_ds 0.1     --augment 0.5     --workers 4     --gpu 0
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


def compute_patch_correspondence(src_centroids, tgt_centroids, gt_q, gt_t, dist_threshold=4.0):
    """
    用 GT 位姿计算 patch 级别的对应关系。

    这是修复对比学习损失的关键函数。原来的代码假设 src 第 i 个 patch 对应 tgt 第 i 个
    patch（对角线假设），但 src 和 tgt 各自独立做 Z-order 排序，这个假设是错误的。
    正确做法：把 src centroids 用 GT 变换到 tgt 坐标系，再找最近邻 tgt patch。

    四元数约定：[x, y, z, w]，w 在最后（scalar-last，与 scipy/KITTI 一致）。

    Args:
        src_centroids:  [B, L, 3]  src 的 patch 中心点，来自 output['src_centroids']
        tgt_centroids:  [B, L, 3]  tgt 的 patch 中心点，来自 output['tgt_centroids']
        gt_q:           [B, 4]     GT 四元数 [x, y, z, w]（cuda tensor）
        gt_t:           [B, 3]     GT 平移向量（cuda tensor）
        dist_threshold: float      最大匹配距离（米），超出则视为无有效对应（非重叠 patch）
                                   KITTI 场景下 patch stride ≈ 16 × voxel_size ≈ 4.8m，
                                   建议设为 4.0（保守），避免把相邻 patch 误判为对应

    Returns:
        correspondence: [B, L] LongTensor  每个 src patch 对应的 tgt patch 索引
        valid_mask:     [B, L] BoolTensor   True = 该 patch 有有效对应（在重叠区域内）
    """
    B, L, _ = src_centroids.shape
    device = src_centroids.device

    # dtype 对齐：gt_q/gt_t 来自 DataLoader（float32），但 model 可能用 fp16
    # 确保旋转矩阵 R 和变换结果与 src_centroids 的 dtype 一致，避免 bmm 类型不匹配
    dtype = src_centroids.dtype
    gt_q = gt_q.to(dtype=dtype)
    gt_t = gt_t.to(dtype=dtype)

    # 1. 四元数 [x, y, z, w] 转旋转矩阵（批量，纯 torch，无需 numpy）
    q = torch.nn.functional.normalize(gt_q, p=2, dim=-1)  # [B, 4]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros(B, 3, 3, device=device, dtype=src_centroids.dtype)
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - z*w)
    R[:, 0, 2] = 2 * (x*z + y*w)
    R[:, 1, 0] = 2 * (x*y + z*w)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - x*w)
    R[:, 2, 0] = 2 * (x*z - y*w)
    R[:, 2, 1] = 2 * (y*z + x*w)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    # R: [B, 3, 3]

    # 2. 把 src centroids 变换到 tgt 坐标系
    #    src_in_tgt = R @ src^T + t  =>  src @ R^T + t
    src_transformed = torch.bmm(src_centroids, R.transpose(1, 2)) + gt_t.unsqueeze(1)
    # src_transformed: [B, L, 3]

    # 3. 计算变换后的 src centroids 到所有 tgt centroids 的距离
    dist = torch.cdist(src_transformed, tgt_centroids)  # [B, L, L]

    # 4. 每个 src patch 找最近的 tgt patch
    min_dist, correspondence = dist.min(dim=-1)  # [B, L]

    # 5. 超出距离阈值的 patch 视为非重叠（无有效对应）
    valid_mask = min_dist < dist_threshold  # [B, L] bool

    return correspondence, valid_mask


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
            args.voxel_size,args.data_list, args.patch_size, args.stride, args.augment
        )
        val_dataset = KittiDataset(
            args.lidar_root, 'val', args.num_points,
            args.voxel_size,args.data_list, args.patch_size, args.stride, augment=0.0
        )
    elif args.dataset == 'nuscenes':
        train_dataset = NuscenesDataset(
            args.lidar_root, 'train', args.num_points,
            args.voxel_size, args.data_list, args.patch_size, args.stride, args.augment
        )
        val_dataset = NuscenesDataset(
            args.lidar_root, 'val', args.num_points,
            args.voxel_size, args.data_list, args.patch_size, args.stride, augment=0.0
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
    shuffle=True,
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
        use_overlap_loss=True,
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
    # 使用 CosineAnnealingLR 替代 StepLR：
    # StepLR 每隔固定步长乘以 gamma（如 0.5），会导致 LR 指数崩塌，
    # 100 轮后 LR 可低至 1e-9，模型参数实际冻结。
    # CosineAnnealingLR 在 T_max 轮内平滑衰减到 eta_min，保证后期仍有学习能力。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epoch,   # 总训练轮数
        eta_min=1e-6,           # 最低学习率，保证训练末期仍有梯度
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
            gt_overlap = batch_data['gt_overlap'].cuda().float()  # [B, L]，必须float，BCE不接受bool/long
            
            # ===== 前向传播 =====
            output = model(src_points, tgt_points)
            # output 包含：
            #   quaternion [B,4], translation [B,3]
            #   src_intermediate_feats: List[Tensor[B,L,D]]  (4层Mamba中间特征)
            #   tgt_intermediate_feats: List[Tensor[B,L,D]]
            #   src_centroids [B,L,3], tgt_centroids [B,L,3]  (patch中心点)

            # ===== 计算 GT patch 对应关系（修复对比学习的核心）=====
            # 原来的代码没有传 correspondence，InfoNCELoss 退化为对角线假设：
            # 假设 src[i] 对应 tgt[i]，但 src/tgt 各自独立 Z-order 排序，
            # 这个假设在有旋转的情况下完全错误，导致 Feature Loss 始终在随机猜测水平。
            # 现在用 GT 位姿把 src centroids 变换到 tgt 坐标系，找真正对应的 patch。
            with torch.no_grad():
                correspondence, valid_mask = compute_patch_correspondence(
                    src_centroids=output['src_centroids'],   # [B, L, 3]
                    tgt_centroids=output['tgt_centroids'],   # [B, L, 3]
                    gt_q=gt_q,                               # [B, 4]，[x,y,z,w] 格式
                    gt_t=gt_t,                               # [B, 3]
                    dist_threshold=4.0,                      # 单位：米，超出视为非重叠区域
                )
            # correspondence: [B, L] LongTensor，每个 src patch 对应的 tgt patch 索引
            # valid_mask:     [B, L] BoolTensor，True = 有有效对应（在重叠区域内）

            # ===== 计算损失 =====
            loss_dict = criterion(
                pred_q=output['quaternion'],
                pred_t=output['translation'],
                gt_q=gt_q,
                gt_t=gt_t,
                overlap_scores=output.get('overlap_scores'),
                src_intermediate=output.get('src_intermediate_feats'),
                tgt_intermediate=output.get('tgt_intermediate_feats'),
                gt_overlap=gt_overlap,
                correspondence=correspondence,  # GT patch 对应索引 [B, L]
                valid_mask=valid_mask,          # 有效对应掩码 [B, L]
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

        # ===== 更新训练历史记录 =====
        history['train_loss'].append(avg_loss)
        history['rot_loss'].append(avg_rot_loss)
        history['trans_loss'].append(avg_trans_loss)
        history['learning_rate'].append(current_lr)

        
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
        val_epochs = list(range(args.eval_interval, len(history['train_loss']) + 1, args.eval_interval))

        # ===== 验证 =====
        if (epoch + 1) % args.eval_interval == 0:
            val_results = validate(model, val_loader, criterion, logger, args, excel_logger, epoch + 1)

            # 记录验证指标到历史
            history['val_loss'].append(val_results['loss'])
            history['val_rot_error'].append(val_results['rot_error'])
            history['val_trans_error'].append(val_results['trans_error'])
            history['val_recall'].append(val_results['recall'])

            
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
                val_epochs=val_epochs,   
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
