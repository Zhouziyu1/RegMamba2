import re
import pandas as pd
import matplotlib.pyplot as plt
import chardet
import numpy as np

# ==================== 全局学术风格设置 ====================
plt.rcParams.update({
    # 字体
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
    'font.size': 10,
    # 坐标轴
    'axes.linewidth': 1.0,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.direction': 'in',          # 刻度朝内
    'ytick.direction': 'in',
    'xtick.major.size': 4,           # 主刻度长度
    'ytick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    # 图例
    'legend.fontsize': 9,
    'legend.frameon': False,         # 无边框
    # 线条
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
    # 网格（默认关闭，若需要可取消注释）
    # 'axes.grid': True,
    # 'grid.color': 'lightgray',
    # 'grid.linestyle': '--',
    # 'grid.alpha': 0.6,
})

# ----------------------------- 自动检测文件编码 -----------------------------
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'

# ----------------------------- 解析日志 -----------------------------
def parse_log(file_path):
    train_records = []
    valid_records = []
    current_epoch = None

    encoding = detect_encoding(file_path)
    print(f"检测到文件编码：{encoding}")

    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 训练行
            m_train = re.search(r'Epoch (\d+): loss=([\d.]+), rot=([\d.]+), trans=([\d.]+)', line)
            if m_train:
                current_epoch = int(m_train.group(1))
                train_records.append({
                    'epoch': current_epoch,
                    'train_loss': float(m_train.group(2)),
                    'train_rot': float(m_train.group(3)),
                    'train_trans': float(m_train.group(4))
                })
                continue

            # 学习率行
            m_lr = re.search(r'Learning rate: ([\d.]+)', line)
            if m_lr and train_records and train_records[-1]['epoch'] == current_epoch:
                train_records[-1]['lr'] = float(m_lr.group(1))
                continue

            # 验证行
            m_valid = re.search(r'验证: loss=([\d.]+), rot_err=([\d.]+)°, trans_err=([\d.]+)m', line)
            if m_valid and current_epoch is not None:
                valid_records.append({
                    'epoch': current_epoch,
                    'valid_loss': float(m_valid.group(1)),
                    'valid_rot_err': float(m_valid.group(2)),
                    'valid_trans_err': float(m_valid.group(3))
                })

    df_train = pd.DataFrame(train_records)
    df_valid = pd.DataFrame(valid_records)

    print(f"解析到训练记录：{len(df_train)} 条")
    print(f"解析到验证记录：{len(df_valid)} 条")

    if df_train.empty:
        raise ValueError("未从日志文件中解析到任何训练记录，请检查日志格式是否匹配。")

    if df_valid.empty:
        print("警告：未找到验证记录，将仅绘制训练曲线。")
        df = df_train.copy()
    else:
        df = pd.merge(df_train, df_valid, on='epoch', how='outer').sort_values('epoch').reset_index(drop=True)

    return df

# ----------------------------- 学术风格绘图 -----------------------------
def plot_metrics(df, save_name='training_curves'):
    """
    绘制符合期刊论文要求的训练/验证曲线
    保存 PNG + PDF 两种格式
    """
    has_valid = 'valid_loss' in df.columns and df['valid_loss'].notna().any()

    # --- 色盲友好配色 (ColorBrewer Set1) ---
    colors = {
        'loss': '#377eb8',      # 蓝
        'rot': '#ff7f00',       # 橙
        'trans': '#4daf4a',     # 绿
        'valid_loss': '#984ea3',# 紫
        'valid_rot': '#e41a1c', # 红
        'valid_trans': '#a65628'# 棕
    }

    # 创建图形布局
    if has_valid:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        fig.suptitle('Training and Validation Metrics', fontsize=14, fontweight='bold')
        ax_train = axes[0]
        ax_valid = axes[1]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        fig.suptitle('Training Metrics', fontsize=14, fontweight='bold')
        ax_train = axes
        ax_valid = None

    # ----- 训练曲线 -----
    # Loss
    ax_train[0].plot(df['epoch'], df['train_loss'], 'o-', 
                     color=colors['loss'], markeredgecolor='white', markeredgewidth=0.5,
                     label='Train Loss')
    ax_train[0].set_xlabel('Epoch')
    ax_train[0].set_ylabel('Loss')
    ax_train[0].set_title('(a) Training Loss')
    ax_train[0].legend(loc='best')

    # Rotation Error
    ax_train[1].plot(df['epoch'], df['train_rot'], 'o-',
                     color=colors['rot'], markeredgecolor='white', markeredgewidth=0.5,
                     label='Train Rot')
    ax_train[1].set_xlabel('Epoch')
    ax_train[1].set_ylabel('Rotation Error (°)')   # 增加单位
    ax_train[1].set_title('(b) Training Rotation Error')
    ax_train[1].legend(loc='best')

    # Translation Error
    ax_train[2].plot(df['epoch'], df['train_trans'], 'o-',
                     color=colors['trans'], markeredgecolor='white', markeredgewidth=0.5,
                     label='Train Trans')
    ax_train[2].set_xlabel('Epoch')
    ax_train[2].set_ylabel('Translation Error (m)') # 增加单位
    ax_train[2].set_title('(c) Training Translation Error')
    ax_train[2].legend(loc='best')

    # ----- 验证曲线（若存在）-----
    if has_valid:
        valid_mask = df['valid_loss'].notna()
        epochs_valid = df.loc[valid_mask, 'epoch']

        # Validation Loss
        ax_valid[0].plot(epochs_valid, df.loc[valid_mask, 'valid_loss'], 's--',
                         color=colors['valid_loss'], markeredgecolor='white', markeredgewidth=0.5,
                         label='Valid Loss')
        ax_valid[0].set_xlabel('Epoch')
        ax_valid[0].set_ylabel('Loss')
        ax_valid[0].set_title('(d) Validation Loss')
        ax_valid[0].legend(loc='best')

        # Validation Rotation Error
        ax_valid[1].plot(epochs_valid, df.loc[valid_mask, 'valid_rot_err'], 's--',
                         color=colors['valid_rot'], markeredgecolor='white', markeredgewidth=0.5,
                         label='Valid Rot')
        ax_valid[1].set_xlabel('Epoch')
        ax_valid[1].set_ylabel('Rotation Error (°)')
        ax_valid[1].set_title('(e) Validation Rotation Error')
        ax_valid[1].legend(loc='best')

        # Validation Translation Error
        ax_valid[2].plot(epochs_valid, df.loc[valid_mask, 'valid_trans_err'], 's--',
                         color=colors['valid_trans'], markeredgecolor='white', markeredgewidth=0.5,
                         label='Valid Trans')
        ax_valid[2].set_xlabel('Epoch')
        ax_valid[2].set_ylabel('Translation Error (m)')
        ax_valid[2].set_title('(f) Validation Translation Error')
        ax_valid[2].legend(loc='best')

    # 调整子图间距
    plt.tight_layout(pad=1.5, w_pad=2.0, h_pad=2.0)
    
    # 保存高分辨率图片（PNG + PDF）
    plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_name}.pdf', dpi=300, bbox_inches='tight')
    print(f"图表已保存为: {save_name}.png 和 {save_name}.pdf")
    plt.show()

def plot_lr(df, save_name='learning_rate'):
    """绘制学习率曲线（学术风格）"""
    if 'lr' in df.columns and df['lr'].notna().any():
        plt.figure(figsize=(6, 3.5))
        plt.plot(df['epoch'], df['lr'], 'o-', color='dimgray', 
                 markeredgecolor='white', markeredgewidth=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.tight_layout()
        plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_name}.pdf', dpi=300, bbox_inches='tight')
        print(f"学习率曲线已保存为: {save_name}.png 和 {save_name}.pdf")
        plt.show()
    else:
        print("日志中未找到学习率信息，跳过学习率曲线。")

# ----------------------------- 主程序 -----------------------------
if __name__ == '__main__':
    log_file = 'train_log.txt'   # 修改为你的日志文件路径
    try:
        df = parse_log(log_file)
        print("\n解析成功！数据预览（前5行）：")
        print(df.head())
        print("\n数据统计信息：")
        print(df.describe())

        plot_metrics(df)
        plot_lr(df)

    except Exception as e:
        print(f"发生错误：{e}")
        print("\n可能的原因及解决方法：")
        print("1. 日志文件路径不正确 → 请将脚本放在与 train_log.txt 同一目录，或填写绝对路径。")
        print("2. 日志格式与正则表达式不匹配 → 请复制几行日志内容，检查是否包含 'Epoch', 'Learning rate', '验证' 等关键词。")
        print("3. 文件编码问题 → 已尝试自动检测，仍失败时可手动指定 encoding='gbk' 或 'utf-8'。")