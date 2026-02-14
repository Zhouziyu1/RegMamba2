# 导入基础数值计算库
import numpy as np
# 导入PyTorch核心库及数据集抽象类（必须继承Dataset实现自定义数据集）
import torch
from torch.utils.data import Dataset

# 导入操作系统路径处理库
import os
# 导入MinkowskiEngine（稀疏卷积库，用于点云体素化/量化）
import MinkowskiEngine as ME

# 导入自定义工具函数：生成随机旋转矩阵（用于数据增强）
from tools.utils import generate_rand_rotm

def read_kitti_bin_voxel(filename, npoints=None, voxel_size=None) -> np.ndarray:
    """
    读取KITTI格式的.bin点云文件，并进行体素下采样、点数筛选和距离/高度过滤
    
    参数:
        filename (str): KITTI点云.bin文件路径（velodyne激光雷达数据）
        npoints (int, optional): 目标保留的点云数量，None则返回全部点
        voxel_size (float, optional): 体素大小（米），None则不进行体素下采样
    
    返回:
        np.ndarray: 处理后的点云数据，形状为[N, 3]，dtype=float32（仅保留xyz坐标）
    """
    # 1. 读取.bin文件：KITTI点云格式为float32，每行4个值（x,y,z,intensity），reshape为[N,4]
    scan = np.fromfile(filename, dtype=np.float32, count=-1).reshape([-1,4])
    # 2. 仅保留xyz坐标，丢弃强度值（intensity）
    scan = scan[:,:3]

    # 3. 体素化下采样（稀疏量化）：将点云按voxel_size划分体素，每个体素保留一个点
    if voxel_size is not None:
        # ME.utils.sparse_quantize：对归一化后的点云量化，返回量化坐标和保留点的索引
        _, sel = ME.utils.sparse_quantize(scan / voxel_size, return_index=True)
        scan = scan[sel]  # 保留体素化后的点云
    
    # 4. 按目标点数筛选（若指定npoints）
    if npoints is None:
        return scan.astype('float32')  # 未指定则返回全部点
    
    # 计算每个点到原点的欧式距离（L2范数）
    dist = np.linalg.norm(scan, ord=2, axis=1)
    N = scan.shape[0]
    # 若点云数量≥目标数：按距离升序排序，保留前npoints个（优先保留近点，信息更丰富）
    if N >= npoints:
        sample_idx = np.argsort(dist)[:npoints]
        scan = scan[sample_idx, :].astype('float32')
        dist = dist[sample_idx]  # 更新距离数组（用于后续过滤）
    else:
        # 点数不足：有放回随机采样至npoints
        idx = np.random.choice(N, npoints, replace=True)
        scan = scan[idx].astype('float32')
        dist = dist[idx]  # 更新距离数组（用于后续过滤）
    
    # 5. 距离+高度过滤：保留距离>3米（过滤过近噪声）且z坐标>-3.5米（过滤地面/过低点）的点
    # scan = scan[np.logical_and(dist > 3., scan[:, 2] > -3.5)]
    return scan

class KittiDataset(Dataset):
    """
    KITTI点云配准数据集类（继承PyTorch Dataset），用于加载点云对和对应的相对位姿
    
    核心逻辑：
    1. 读取序列的配对文件（.txt），解析源/目标点云路径和相对位姿
    2. 加载点云并进行预处理（体素下采样、点数筛选）
    3. 可选数据增强（随机旋转目标点云）
    4. 返回张量格式的源点云、目标点云、相对位姿
    """
    def __init__(self, root, seqs, npoints, voxel_size, data_list, augment=0.0):
        """
        初始化数据集
        
        参数:
            root (str): KITTI数据集根路径（包含各序列文件夹，如00/velodyne）
            seqs (str): 数据集划分标识（如 'train'/'val'），用于拼接生成数据列表的文件名（如 train.txt）
            npoints (int): 每个点云保留的目标点数
            voxel_size (float): 体素大小（用于点云下采样）
            data_list (str): 存放序列配对和位姿的文件夹路径
            augment (float, optional): 数据增强概率（0~1），0则不增强
        """
        super(KittiDataset, self).__init__()
        self.root = root          # 数据集根路径
        self.seqs = seqs          # 序列名（如'train'）
        self.npoints = npoints    # 目标点数
        self.voxel_size = voxel_size  # 体素大小
        self.augment = augment    # 数据增强概率
        self.data_list = data_list    # 配对/位姿文件路径
        # 构建数据集列表（包含所有点云对和位姿的字典）
        self.dataset = self.make_dataset()
    
    def make_dataset(self):
        """
        构建数据集列表：解析序列的配对文件，生成包含源/目标点云路径、相对位姿的字典列表
        
        返回:
            list[dict]: 每个元素为{'points1': 源点云路径, 'points2': 目标点云路径, 'Tr': 4×4相对位姿矩阵}
        """
        # 构造4×4齐次变换矩阵的最后一行（[0,0,0,1]），用于补全3×4位姿为4×4
        last_row = np.zeros((1,4), dtype=np.float32)
        last_row[:,3] = 1.0
        dataset = []  # 存储所有数据对

        # 拼接配对+位姿文件路径（如data_list/00.txt）
        fn_pair_poses = os.path.join(self.data_list, self.seqs + '.txt')
        # 读取文件：每行15个值（序列号、源帧号、目标帧号、3×4位姿的12个值），reshape为[N,15]
        metadata = np.genfromtxt(fn_pair_poses).reshape([-1, 15])
        
        # 遍历所有数据对
        for i in range(metadata.shape[0]):
            folder = os.path.join(self.root, '%02d'%metadata[i][0], 'velodyne')
            src_fn = os.path.join(folder, '%06d.bin'%metadata[i][1])
            dst_fn = os.path.join(folder, '%06d.bin'%metadata[i][2])
            rela_pose = metadata[i][3:].reshape(3,4).astype(np.float32)
            rela_pose = np.concatenate([rela_pose, last_row], axis = 0)
            data_dict = {'points1': src_fn, 'points2': dst_fn, 'Tr': rela_pose}
            dataset.append(data_dict)

        return dataset
    
    def __getitem__(self, index):
        """
        按索引读取单个数据对（必须实现的方法）
        
        参数:
            index (int): 数据索引
        
        返回:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - src_points: 源点云 [N,3]
                - dst_points: 目标点云 [M,3]
                - Tr: 源→目标的相对位姿 [4,4]
        """
        # 1. 读取数据字典（包含路径和位姿）
        data_dict = self.dataset[index]
        # 2. 加载并预处理源/目标点云
        src_points = read_kitti_bin_voxel(data_dict['points1'], self.npoints, self.voxel_size)
        dst_points = read_kitti_bin_voxel(data_dict['points2'], self.npoints, self.voxel_size)
        Tr = data_dict['Tr'].copy()  # 4×4相对位姿矩阵
        
        # 3. 数据增强：以augment概率随机旋转目标点云（增强模型泛化性）
        # 数据增强（仅训练时）
        if self.augment > 0 and np.random.rand() < self.augment:
            aug_T = np.eye(4, dtype=np.float32)
            aug_T[:3,:3] = generate_rand_rotm(1.0, 1.0)
            dst_points = dst_points @ aug_T[:3,:3]
            Tr = Tr @ aug_T

        # 旋转矩阵 → 四元数
        from scipy.spatial.transform import Rotation as R
        quat = R.from_matrix(Tr[:3,:3]).as_quat()   # [x,y,z,w]

        return {
            'src_points': torch.from_numpy(src_points).float(),
            'tgt_points': torch.from_numpy(dst_points).float(),
            'gt_quaternion': torch.from_numpy(quat).float(),
            'gt_translation': torch.from_numpy(Tr[:3,3]).float(),
            'Tr': torch.from_numpy(Tr).float(),
        }

    def __len__(self):
        return len(self.dataset)