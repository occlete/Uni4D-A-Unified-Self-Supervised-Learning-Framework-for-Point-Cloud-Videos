import os
import sys
import numpy as np
import random
import open3d as o3d
from pyquaternion import Quaternion
from torch.utils.data import Dataset
import h5py


def get_mapping(mapping_file):
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    return actions_dict


class SegDataset(Dataset):
    def __init__(self, root=None, train=True):
        super(SegDataset, self).__init__()

        self.train = train

        # 存储所有文件路径
        self.files = []
        if self.train:
            filenames = ['train1.h5', 'train2.h5', 'train3.h5', 'train4.h5']
        else:
            filenames = ['test1.h5', 'test2.h5']

        # 保存文件路径
        for filename in filenames:
            self.files.append(root + '/' + filename)

        # 计算总样本数
        self.sample_count = []
        for file_path in self.files:
            with h5py.File(file_path, 'r') as f:
                self.sample_count.append(len(f['pcd']))

        # 计算所有文件的累积样本数，用于索引映射
        self.cumulative_count = np.cumsum(self.sample_count)

    def __len__(self):
        # 返回数据集的总样本数
        return self.cumulative_count[-1]  # 最后一个文件的总样本数

    def augment(self, pc, center):
        flip = np.random.uniform(0, 1) > 0.5
        if flip:
            pc = (pc - center)
            pc[:, 0] *= -1
            pc += center
        else:
            pc = pc - center
            jittered_data = np.clip(0.01 * np.random.randn(150, 2048, 3), -0.05, 0.05)
            jittered_data += pc
            pc = pc + center

        scale = np.random.uniform(0.8, 1.2)
        pc = (pc - center) * scale + center

        return pc

    def __getitem__(self, index):
        # 通过二分查找找到当前样本属于哪个文件
        file_idx = np.searchsorted(self.cumulative_count, index, side='right')

        # 获取当前样本在文件中的索引
        if file_idx > 0:
            index_in_file = index - self.cumulative_count[file_idx - 1]
        else:
            index_in_file = index

        file_path = self.files[file_idx]

        with h5py.File(file_path, 'r') as f:
            pc = f['pcd'][index_in_file]  # 获取点云数据
            center_0 = f['center'][index_in_file][0]  # 获取中心点
            label = f['label'][index_in_file]  # 获取标签
             # 对点云数据进行降采样（从2048个点中随机选取1024个点）
            if pc.shape[0] > 1024:
                idx = np.random.choice(pc.shape[0], 1024, replace=False)  # 随机选择1024个点的索引
                pc = pc[idx]  # 降采样后的点云数据
                center_0 = center_0[idx]
                
            if self.train:
                pc = self.augment(pc, center_0)

        return pc.astype(np.float32)#, label.astype(np.int64)


if __name__ == '__main__':
    datasets = SegDataset(root='/mnt/hdd/ZuoZhi/ActionSeg', train=False)

